# warm_start.py
# Multi-site warm-start calibration:
# - Global Sobol seeds across sites
# - Global GP with product kernel (params x site features)
# - Save/reuse seed (X, Y) per site
# - Per-site TR-TS with warm prior from global GP

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import shutil
import time
import pandas as pd
import sys
# from within environment_calibration_common submodule
from helpers import load_coordinator_df
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel, ProductKernel
from gpytorch.means import Mean, ConstantMean
from torch.quasirandom import SobolEngine
from translate_parameters import translate_parameters, get_initial_samples
from run_sims import submit_sim
from get_eradication import get_eradication
from compare_to_data.run_full_comparison import compute_scores_across_site
sys.path.append("../simulations")
import manifest as manifest
from turbo_thompson_sampling import TurboThompsonSampling
import run_simulation_for_site

# ──────────────────────────────────────────────────────────────────────────────
# 0) USER SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

# Four habitat multipliers with explicit (min, max) ranges
param_ranges: Dict[str, Tuple[float, float]] = {
    "hab_mult_1": (0.10, 2.0),
    "hab_mult_2": (0.05, 1.5),
    "hab_mult_3": (0.20, 3.0),
    "hab_mult_4": (0.01, 1.0),
}

# Provide site features (start with lat/lon; later replace via CSV loader)
# Map site -> feature vector (any numeric features). Keep length F consistent.
site_features: Dict[str | int, List[float]] = {
    101: [0.12, 36.48],   # (lat, lon) example
    205: [0.35, 36.70],
    309: [0.62, 36.91],
    412: [0.77, 36.12],
}

# Which sites to include in global (multi-site) MTGP phase
global_sites: List[str | int] = [101, 205, 309]


# globals for now–
exp_label = manifest.EXPERIMENT_LABEL
output_dir = f"output/{exp_label}"
calib_coord_path = manifest.calibration_coordinator_path
param_key_path = "parameter_key.csv"
gs_coord_path =  manifest.global_simulation_coordinator_path
weights_path = "simulation_inputs/weights.csv"



def f_sim(X_real: np.ndarray, params: dict, site) -> float:
    
    
   Y_scores = run_simulation_for_site(site, exp_label, output_dir, calib_coord_path, param_key_path, gs_coord_path, weights_path)
    
   return Y_scores["total_score"]





# ──────────────────────────────────────────────────────────────────────────────
# 1) UTILITIES: normalization, sobol, I/O
# ──────────────────────────────────────────────────────────────────────────────

def ranges_to_tensors(ranges: Dict[str, Tuple[float, float]],
                      device="cpu", dtype=torch.double) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    names = list(ranges.keys())
    bounds = np.array([ranges[k] for k in names], dtype=float)
    lb = torch.tensor(bounds[:, 0], dtype=dtype, device=device)
    ub = torch.tensor(bounds[:, 1], dtype=dtype, device=device)
    return lb, ub, names

def unnormalize(X_unit: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    return lb + (ub - lb) * X_unit

def normalize_features(features: Dict[str | int, List[float]],
                       only_sites: Optional[List[str | int]] = None,
                       device="cpu", dtype=torch.double) -> Tuple[Dict[str | int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Min-max normalize site features to [0,1] feature-wise over the provided sites."""
    keys = only_sites if only_sites is not None else list(features.keys())
    M = np.array([features[k] for k in keys], dtype=float)
    F = M.shape[1]
    fmin = torch.tensor(M.min(axis=0), dtype=dtype, device=device)
    fmax = torch.tensor(M.max(axis=0), dtype=dtype, device=device)
    fspan = torch.clamp(fmax - fmin, min=1e-8)
    out = {}
    for k in keys:
        v = torch.tensor(features[k], dtype=dtype, device=device)
        out[k] = torch.clamp((v - fmin) / fspan, 0.0, 1.0)
    return out, fmin, fmax

def sobol_unit(n: int, d: int, seed: Optional[int] = None,
               device="cpu", dtype=torch.double) -> torch.Tensor:
    eng = SobolEngine(dimension=d, scramble=True, seed=seed)
    X = eng.draw(n=n).to(device=device, dtype=dtype)
    return X

def save_site_seed(site, X_unit: torch.Tensor, Y: torch.Tensor, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    torch.save(X_unit.cpu(), os.path.join(outdir, f"seed_X_site{site}.pt"))
    torch.save(Y.cpu(),      os.path.join(outdir, f"seed_Y_site{site}.pt"))

def load_site_seed(site, outdir: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    xpth = os.path.join(outdir, f"seed_X_site{site}.pt")
    ypth = os.path.join(outdir, f"seed_Y_site{site}.pt")
    if os.path.exists(xpth) and os.path.exists(ypth):
        return torch.load(xpth), torch.load(ypth)
    return None

# ──────────────────────────────────────────────────────────────────────────────
# 2) PROBLEM WRAPPER: converts [0,1]^D -> real units; caches seeds to skip sims
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SiteProblem:
    site: str | int
    param_ranges: Dict[str, Tuple[float, float]]
    params: dict
    # Optional cache: map tuple(X_unit) -> y to avoid re-simulating seeds
    cache: Optional[Dict[Tuple[float, ...], float]] = None
    device: str = "cpu"
    dtype: torch.dtype = torch.double

    def __post_init__(self):
        self.lb, self.ub, self.names = ranges_to_tensors(self.param_ranges, self.device, self.dtype)
        self.dim = len(self.names)

    def __call__(self, X_unit: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X_unit = X_unit.to(device=self.device, dtype=self.dtype)
        X_real = unnormalize(X_unit, self.lb, self.ub).cpu().numpy()
        Ys: List[float] = []
        for i in range(X_unit.size(0)):
            key = tuple(np.round(X_unit[i].cpu().numpy(), 10).tolist())
            if self.cache is not None and key in self.cache:
                y = self.cache[key]
            else:
                y = float(f_sim(X_real[i], self.params, self.site))
                if self.cache is not None:
                    self.cache[key] = y
            Ys.append(y)
        Y = torch.tensor(Ys, dtype=self.dtype, device=self.device)
        # Return the *normalized* X back (optimizer stores unit space)
        return X_unit, Y

# ──────────────────────────────────────────────────────────────────────────────
# 3) GLOBAL PHASE: evaluate shared Sobol seeds on multiple sites & save
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_global_seeds(sites: List[str | int],
                          N_seed: int,
                          outdir: str,
                          seed: Optional[int] = 123,
                          params: Optional[dict] = None,
                          device="cpu", dtype=torch.double) -> Dict[str | int, Dict[str, torch.Tensor]]:
    """Run the same Sobol seeds on each site; save and return per-site data."""
    params = params or {}
    D = len(param_ranges)
    X_unit_shared = sobol_unit(N_seed, D, seed=seed, device=device, dtype=dtype)

    results = {}
    for site in sites:
        # Load if already saved
        loaded = load_site_seed(site, outdir)
        if loaded is not None:
            X_u, Y = loaded
        else:
            problem = SiteProblem(site=site, param_ranges=param_ranges, params=params,
                                  cache={}, device=device, dtype=dtype)
            X_u, Y = problem(X_unit_shared)
            save_site_seed(site, X_u, Y, outdir)
        results[site] = {"X": X_u.to(device=device, dtype=dtype),
                         "Y": Y.to(device=device, dtype=dtype)}
    return results

# ──────────────────────────────────────────────────────────────────────────────
# 4) GLOBAL GP: product kernel on [params] × [site features]
# ──────────────────────────────────────────────────────────────────────────────

class PriorMean(Mean):
    """Wraps a callable μ(x) -> (B,) into a GPyTorch Mean module."""
    def __init__(self, fn):  # fn: (B,D) -> (B,)
        super().__init__()
        self.fn = fn
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)

def fit_global_gp(global_cache: Dict[str | int, Dict[str, torch.Tensor]],
                  features: Dict[str | int, List[float]],
                  only_sites: Optional[List[str | int]] = None,
                  device="cpu", dtype=torch.double):
    """Build X_aug = [X_unit, site_features_norm] and fit SingleTaskGP with product kernel."""
    keys = only_sites if only_sites is not None else list(global_cache.keys())
    # Normalize site features over the subset
    feats_norm, fmin, fmax = normalize_features(features, keys, device=device, dtype=dtype)
    # Build augmented training set
    X_blocks, Y_blocks = [], []
    for s in keys:
        Xs = global_cache[s]["X"]  # (N_seed, D) in [0,1]
        Ys = global_cache[s]["Y"].unsqueeze(-1)  # (N_seed,1)
        Fs = feats_norm[s].expand(Xs.size(0), -1)  # (N_seed, F)
        X_aug = torch.cat([Xs, Fs], dim=-1)  # (N_seed, D+F)
        X_blocks.append(X_aug)
        Y_blocks.append(Ys)
    train_X = torch.cat(X_blocks, dim=0)  # (N_seed*|sites|, D+F)
    train_Y = torch.cat(Y_blocks, dim=0)  # (N_seed*|sites|, 1)

    D = global_cache[keys[0]]["X"].size(1)
    F = feats_norm[keys[0]].numel()

    # Product kernel: K = K_param(x,x') * K_site(f,f')
    k_param = RBFKernel(ard_num_dims=D, active_dims=torch.arange(0, D))
    k_site  = RBFKernel(ard_num_dims=F, active_dims=torch.arange(D, D + F))
    covar   = ScaleKernel(ProductKernel(k_param, k_site))
    mean    = ConstantMean()

    model = SingleTaskGP(train_X=train_X, train_Y=train_Y,
                         covar_module=covar, mean_module=mean)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    model.eval()
    return model, feats_norm, (fmin, fmax)

def make_site_prior_fn(global_gp: SingleTaskGP,
                       site_feature_norm: torch.Tensor,
                       D: int, device="cpu", dtype=torch.double):
    """μ_site(x_unit) from global GP, using normalized site feature vector."""
    site_feature_norm = site_feature_norm.to(device=device, dtype=dtype)
    def μ(x_query: torch.Tensor) -> torch.Tensor:
        x_query = x_query.to(device=device, dtype=dtype)  # (B, D)
        B = x_query.size(0)
        F = site_feature_norm.numel()
        Fs = site_feature_norm.expand(B, F)
        X_aug = torch.cat([x_query, Fs], dim=-1)
        with torch.no_grad():
            post = global_gp.posterior(X_aug)
            return post.mean.squeeze(-1)  # (B,)
    return μ

# ──────────────────────────────────────────────────────────────────────────────
# 5) LOCAL (PER-SITE) TRUST-REGION TS with warm prior & cached seeds
# ──────────────────────────────────────────────────────────────────────────────

def run_site_turbo(site,
                   cached: Dict[str, torch.Tensor],
                   prior_fn,  # callable μ(x_unit) or None
                   n_iter: int = 20,
                   batch_size: int = 4,
                   n_candidates: int = 2000,
                   device="cpu", dtype=torch.double):
    """Runs per-site TR-TS minimizing the score."""
    X_obs = cached["X"].to(device=device, dtype=dtype)          # (N0, D) in [0,1]
    Y_obs = cached["Y"].to(device=device, dtype=dtype).unsqueeze(-1)  # (N0, 1)

    D = X_obs.size(1)
    covar = ScaleKernel(RBFKernel(ard_num_dims=D))
    mean  = PriorMean(prior_fn) if prior_fn is not None else ConstantMean()

    # Initialize Turbo state
    tts = TurboThompsonSampling(dim=D,
                                batch_size=batch_size,
                                failure_tolerance=5,
                                success_tolerance=8,
                                n_candidates=n_candidates)

    for it in range(n_iter):
        # Fit local GP
        gp = SingleTaskGP(train_X=X_obs, train_Y=Y_obs,
                          covar_module=covar, mean_module=mean)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        gp.eval()

        # Propose next batch within trust region
        X_next = tts.generate_batch(gp, X_obs, Y_obs)  # expects unit cube
        # Evaluate simulator in REAL units using SiteProblem cache to avoid dupes
        problem = SiteProblem(site=site, param_ranges=param_ranges, params={},
                              cache={}, device=device, dtype=dtype)
        _, Y_next_flat = problem(X_next)  # (B,)
        Y_next = Y_next_flat.unsqueeze(-1)

        # Update observations and Turbo state
        X_obs = torch.cat([X_obs, X_next], dim=0)
        Y_obs = torch.cat([Y_obs, Y_next], dim=0)
        tts.update(X_obs, Y_obs)

        # Optional: stopping condition
        if getattr(tts, "stopping_condition", False):
            break

    best_idx = torch.argmin(Y_obs.squeeze(-1)).item()
    best_x_unit = X_obs[best_idx]
    best_y = float(Y_obs[best_idx])
    return best_x_unit, best_y, X_obs, Y_obs

# ──────────────────────────────────────────────────────────────────────────────
# 6) MAIN WORKFLOW
# ──────────────────────────────────────────────────────────────────────────────

def main():
    torch.set_default_dtype(torch.double)
    device = "cpu"

    # (A) GLOBAL PHASE: evaluate shared Sobol seeds across selected sites
    N_seed = 16
    seed_dir = "global_seeds"
    global_cache = evaluate_global_seeds(global_sites, N_seed, seed_dir,
                                         seed=123, params={}, device=device)

    # (B) Fit GLOBAL GP on [params] × [site features] with product kernel
    global_gp, feats_norm, (fmin, fmax) = fit_global_gp(global_cache, site_features,
                                                        only_sites=global_sites, device=device)

    # (C) PER-SITE: warm prior & TR-TS
    results = {}
    for site in global_sites:
        # Build warm prior μ_site(x_unit)
        D = global_cache[site]["X"].size(1)
        site_feat_norm = feats_norm[site]
        prior_fn = make_site_prior_fn(global_gp, site_feat_norm, D=D, device=device)

        best_x_unit, best_y, X_hist, Y_hist = run_site_turbo(
            site=site,
            cached=global_cache[site],
            prior_fn=prior_fn,
            n_iter=20,
            batch_size=4,
            n_candidates=2000,
            device=device,
        )

        # Save per-site results
        os.makedirs("site_results", exist_ok=True)
        torch.save(X_hist.cpu(), f"site_results/X_hist_site{site}.pt")
        torch.save(Y_hist.cpu(), f"site_results/Y_hist_site{site}.pt")
        results[site] = {
            "best_x_unit": best_x_unit.cpu().numpy().tolist(),
            "best_score": best_y,
        }
        print(f"[site {site}] best score = {best_y:.6f} at unit X = {best_x_unit.cpu().numpy()}")

    # Write summary
    with open("site_results/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved site_results/summary.json")

# ──────────────────────────────────────────────────────────────────────────────
# 7) (Optional) Hook into bo.py:initRandom instead of local Sobol
#     If you want to reuse your existing initRandom, you can swap the
#     seed evaluator above (evaluate_global_seeds) with a variant that
#     constructs a BO(problem=SiteProblem(...)) and calls bo.initRandom.
#     The SiteProblem cache ensures repeated seeds are returned without
#     re-simulation, so no change to bo.py is strictly required.
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
