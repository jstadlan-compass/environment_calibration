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
from batch_generators.turbo_thompson_sampling import TurboThompsonSampling 
import run_simulation_for_site

from typing import Dict, Optional, Sequence, Tuple, Callable, Union
import matplotlib
import matplotlib.pyplot as plt

import manifest

# ──────────────────────────────────────────────────────────────────────────────
# 0) USER SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
GLOBAL_ONLY = True


# Four habitat multipliers with explicit (min, max) ranges
param_ranges: Dict[str, Tuple[float, float]] = {
    "hab_mult_1": (0.10, 2.0),
    "hab_mult_2": (0.05, 1.5),
    "hab_mult_3": (0.20, 3.0),
    "hab_mult_4": (0.01, 1.0),
}

# Provide site features (start with lat/lon; later replace via CSV loader)
# Map site -> feature vector (any numeric features). Keep length F consistent.
gs_coord_df = pd.read_csv(manifest.global_simulation_coordinator_path)
site_features: dict[str, list[float]] = dict(
    zip(gs_coord_df["site"], gs_coord_df[["lat", "lon"]].values.tolist())
)

 
# site_features: Dict[str | int, List[float]] = {
#     101: [0.12, 36.48],   # (lat, lon) example
#     205: [0.35, 36.70],
#     309: [0.62, 36.91],
#     412: [0.77, 36.12],
# }

# Which sites to include in global (multi-site) MTGP phase
global_sites = gs_coord_df["site"]


# globals for now–
exp_label = manifest.EXPERIMENT_LABEL
output_dir = f"output/{exp_label}"
calib_coord_path = manifest.calibration_coordinator_path
param_key_path = "parameter_key.csv"
gs_coord_path =  manifest.global_simulation_coordinator_path
weights_path = "simulation_inputs/weights.csv"



def f_sim(X_real: np.ndarray, params: dict, site) -> float:
    
    
   #Y_scores = run_simulation_for_site(site, exp_label, output_dir, calib_coord_path, param_key_path, gs_coord_path, weights_path)
    
   #return Y_scores["total_score"]
   return np.random.rand()




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
# 6) PLOTTING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────


def _torchify_like(module_or_likelihood, X_np):
    """Make a torch tensor on the same device/dtype as the model/likelihood."""
    if torch is None:
        raise TypeError("Torch not available but a torch model was provided.")
    # Try to grab dtype/device from any parameter; default to float32/cpu.
    dtype = torch.float32
    device = torch.device("cpu")
    try:
        p = next(module_or_likelihood.parameters())
        dtype = p.dtype
        device = p.device
    except Exception:
        pass
    return torch.as_tensor(X_np, dtype=dtype, device=device)


def _make_gp_predictor(gp_obj):
    """
    Return a callable f(X_np) -> (mean_np, std_np).
    Supports:
      - (gpytorch_model, likelihood) tuple
      - gpytorch model with .posterior(...)
      - sklearn-like: .predict(X, return_std=True)
      - generic callable: returns (mean, std|var)
    NOTE: We intentionally check GPyTorch cases BEFORE the generic callable branch.
    """
    # Case A: tuple/list (model, likelihood)
    if isinstance(gp_obj, (tuple, list)) and len(gp_obj) >= 2:
        model, likelihood = gp_obj[:2]
        def _call(X_np):
            X_t = _torchify_like(model, X_np)
            model.eval(); likelihood.eval()
            with torch.no_grad():
                preds = likelihood(model(X_t))  # MultivariateNormal
                mean = _to_numpy(preds.mean).reshape(-1)
                var = preds.variance
                if hasattr(var, "to_dense"):
                    var = var.to_dense()
                std = np.sqrt(_to_numpy(var).reshape(-1))
            return mean, std
        return _call

    # Case B: gpytorch model with .posterior (some wrappers expose this)
    if hasattr(gp_obj, "posterior"):
        def _call(X_np):
            X_t = _torchify_like(gp_obj, X_np)
            try:
                gp_obj.eval()
            except Exception:
                pass
            with torch.no_grad():
                post = gp_obj.posterior(X_t)
                mean = _to_numpy(post.mean).reshape(-1)
                var = post.variance
                if hasattr(var, "to_dense"):
                    var = var.to_dense()
                std = np.sqrt(_to_numpy(var).reshape(-1))
            return mean, std
        return _call

    # Case C: sklearn-like
    if hasattr(gp_obj, "predict"):
        def _call(X_np):
            mean, std = gp_obj.predict(X_np, return_std=True)
            return _to_numpy(mean).reshape(-1), _to_numpy(std).reshape(-1)
        return _call

    # Case D: generic callable returning (mean, std|var)
    if callable(gp_obj):
        def _call(X_np):
            out = gp_obj(X_np)
            if not (isinstance(out, tuple) and len(out) >= 2):
                raise ValueError("Callable GP must return (mean, std) or (mean, var).")
            mean, second = out[:2]
            mean = _to_numpy(mean).reshape(-1)
            second = _to_numpy(second).reshape(-1)
            # Assume second is std unless clearly variance
            std = np.sqrt(second) if np.any(second < 0) or np.mean(second) > 1e3 else second
            return mean, std
        return _call

    raise TypeError("Unsupported GP object. Pass (model, likelihood), a model with .posterior, .predict(..., return_std=True), or a callable.")

def _build_partial_grid(
    X_ref: np.ndarray,
    j: int,
    x_min: float,
    x_max: float,
    n_grid: int = 200
) -> np.ndarray:
    """Create a grid along dimension j while holding others at X_ref."""
    xg = np.linspace(x_min, x_max, n_grid)
    Xg = np.tile(X_ref, (n_grid, 1))
    Xg[:, j] = xg
    return Xg, xg


def plot_y_vs_each_x_from_results_with_gp(
    results: Dict[str, Dict[str, object]],
    gp_model_or_predictor: Union[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], object],
    *,
    x_labels: Optional[Sequence[str]] = None,
    title_prefix: str = "Global seed evaluation",
    figsize: Tuple[float, float] = (7.0, 4.8),
    alpha_points: float = 0.9,
    alpha_band: float = 0.20,
    save_dir: Optional[str] = None,
    fmt: str = "png",
    dpi: int = 160,
    n_grid: int = 200,
    x_ref: Optional[Sequence[float]] = None,   # partial dep ref point; default = column medians of all X
    x_bounds: Optional[Sequence[Tuple[float, float]]] = None,  # per-dim (min,max); default from data with small pad
    pad_frac: float = 0.03,
) -> None:
    """
    For each X dimension, plot: scatter of Y vs X[:, j] (colored by site),
    plus GP mean line and 95% CI band from the GP's predictive std.

    results[site] = {"X": (n_i, d), "Y": (n_i,)}
    gp_model_or_predictor: callable or model; must return mean & std for an (n,d) X.
    """
    # --- gather data across sites ---
    site_names, X_chunks, Y_chunks = [], [], []
    d_first = None
    for site, dct in results.items():
        X_i = _to_numpy(dct["X"])
        Y_i = _to_numpy(dct["Y"]).reshape(-1)
        if X_i.ndim != 2:
            raise ValueError(f"results['{site}']['X'] must be 2D, got {X_i.shape}")
        if Y_i.shape[0] != X_i.shape[0]:
            raise ValueError(f"results['{site}'] length mismatch: Y={Y_i.shape[0]} vs X rows={X_i.shape[0]}")
        if d_first is None:
            d_first = X_i.shape[1]
        elif X_i.shape[1] != d_first:
            raise ValueError(f"All sites must share the same X dimensionality; mismatch at site '{site}'.")
        X_chunks.append(X_i); Y_chunks.append(Y_i)
        site_names.extend([site] * X_i.shape[0])

    if d_first is None:
        raise ValueError("Empty results.")

    X_all = np.vstack(X_chunks)        # (N, d)
    Y_all = np.concatenate(Y_chunks)   # (N,)
    sites_all = np.array(site_names)   # (N,)
    d = d_first

    if not x_labels or len(x_labels) != d:
        x_labels = [f"X[{j}]" for j in range(d)]

    # default reference: column medians
    if x_ref is None:
        x_ref = np.median(X_all, axis=0)
    else:
        x_ref = np.asarray(x_ref).reshape(-1)
        if x_ref.shape[0] != d:
            raise ValueError(f"x_ref must have length {d} (one ref value per dimension).")

    # default bounds: from data with small padding
    if x_bounds is None:
        mins = X_all.min(axis=0)
        maxs = X_all.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-9)
        x_bounds = [(mins[j] - pad_frac * spans[j], maxs[j] + pad_frac * spans[j]) for j in range(d)]
    else:
        if len(x_bounds) != d:
            raise ValueError(f"x_bounds must be a sequence of {d} (min,max) tuples.")

    # palette per site
# palette per site (drop-in replacement)
    uniq_sites = np.unique(sites_all)
    try:
        cmap = plt.cm.get_cmap("tab20", max(len(uniq_sites), 3))
    except TypeError:
        # Older Matplotlib without the N parameter
        cmap = plt.cm.get_cmap("tab20")
    color_map = {s: cmap(i % (getattr(cmap, 'N', 20))) for i, s in enumerate(uniq_sites)}
    



    # make predictor
    gp_predict = _make_gp_predictor(gp_model_or_predictor)

    # --- plot per dimension ---
    for j in range(d):
        fig, ax = plt.subplots(figsize=figsize)

        # scatter by site
        for s in uniq_sites:
            m = (sites_all == s)
            ax.scatter(
                X_all[m, j], Y_all[m],
                label=str(s),
                s=28,
                alpha=alpha_points,
                edgecolors="none",
                c=[color_map[s]],
                zorder=3
            )

        # GP partial dependence along dim j
        (x_lo, x_hi) = x_bounds[j]
        Xg, xg = _build_partial_grid(np.array(x_ref, dtype=float), j, x_lo, x_hi, n_grid=n_grid)
        mu, sd = gp_predict(Xg)   # shapes: (n_grid,), (n_grid,)

        ci_lo = mu - 1.96 * sd
        ci_hi = mu + 1.96 * sd

        # shaded CI first (under the line & points)
        ax.fill_between(xg, ci_lo, ci_hi, alpha=alpha_band, linewidth=0, label="95% CI", zorder=1)
        # mean line
        ax.plot(xg, mu, linewidth=2.0, label="GP mean", zorder=2)

        # cosmetics
        ax.set_xlabel(x_labels[j])
        ax.set_ylabel("Y")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        ax.set_title(f"{title_prefix}: Y vs {x_labels[j]}")

        # legend: sites + GP entries
        handles, labels = ax.get_legend_handles_labels()
        # Ensure "GP mean" and "95% CI" appear last & only once
        # (already included via plot/fill_between)
        leg = ax.legend(
            handles, labels,
            title="Legend",
            loc="best",
            frameon=True,
            framealpha=0.9,
            scatterpoints=1,
            markerscale=1.3
        )

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            safe = x_labels[j].replace(' ', '_').replace('[','').replace(']','')
            fig.savefig(os.path.join(save_dir, f"y_vs_{safe}.{fmt}"), dpi=dpi)
            plt.close(fig)
        else:
            plt.show()

def _to_numpy(x):
    """Accept torch.Tensor or array-like; return 1D/2D numpy on CPU."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_y_vs_each_x_from_results(
    results: Dict[str, Dict[str, object]],
    *,
    x_labels: Optional[Sequence[str]] = None,
    title_prefix: str = "Global seed evaluation",
    figsize: Tuple[float, float] = (6.5, 4.5),
    alpha: float = 0.85,
    save_dir: Optional[str] = None,
    fmt: str = "png",
    dpi: int = 160,
) -> None:
    """
    results[site] = {"X": X_u (n_i, d), "Y": Y (n_i,)}, tensors possibly on device.
    Produces d separate figures: scatter of Y vs X[:, j], colored by site.
    """
    # --- collect and validate ---
    site_names = []
    X_chunks = []
    Y_chunks = []

    d_first = None
    for site, dct in results.items():
        if "X" not in dct or "Y" not in dct:
            raise ValueError(f"results['{site}'] must have keys 'X' and 'Y'")

        X_i = _to_numpy(dct["X"])
        Y_i = _to_numpy(dct["Y"]).reshape(-1)

        if X_i.ndim != 2:
            raise ValueError(f"results['{site}']['X'] must be 2D, got {X_i.shape}")
        if Y_i.shape[0] != X_i.shape[0]:
            raise ValueError(
                f"results['{site}'] length mismatch: Y={Y_i.shape[0]} vs X rows={X_i.shape[0]}"
            )

        if d_first is None:
            d_first = X_i.shape[1]
        elif X_i.shape[1] != d_first:
            raise ValueError(
                f"All sites must share the same X dimensionality; got {d_first} and {X_i.shape[1]} for site '{site}'."
            )

        X_chunks.append(X_i)
        Y_chunks.append(Y_i)
        site_names.extend([site] * X_i.shape[0])

    if d_first is None:
        raise ValueError("Empty results: no sites found.")

    X_all = np.vstack(X_chunks)        # (N, d)
    Y_all = np.concatenate(Y_chunks)   # (N,)
    sites_all = np.array(site_names)   # (N,)
    print(sites_all)

    d = d_first
    if not x_labels or len(x_labels) != d:
        x_labels = [f"X[{j}]" for j in range(d)]

    # --- color map per site ---
    uniq_sites = np.unique(sites_all)
    cmap = plt.cm.get_cmap("tab20", max(len(uniq_sites), 3))
    color_map = {s: cmap(i % cmap.N) for i, s in enumerate(uniq_sites)}

    # --- plot: one figure per dimension ---
    for j in range(d):
        fig, ax = plt.subplots(figsize=figsize)
        for s in uniq_sites:
            mask = (sites_all == s)
            ax.scatter(
                X_all[mask, j],
                Y_all[mask],
                label=str(s),
                s=28,
                alpha=alpha,
                edgecolors="none",
                c=[color_map[s]],
            )
        ax.set_xlabel(x_labels[j])
        ax.set_ylabel("Y")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        ax.set_title(f"{title_prefix}: Y vs {x_labels[j]}")
        leg = ax.legend(
            title="Site",
            loc="best",
            frameon=True,
            framealpha=0.9,
            scatterpoints=1,   # for scatter legends
            markerscale=1.4,   # scales marker size in legend
        )
            
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"y_vs_{x_labels[j].replace(' ', '_').replace('[','').replace(']','')}.{fmt}"
            fig.savefig(os.path.join(save_dir, fname), dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
            

def _eval_prior_fn(prior_fn, X_t):
    """
    Run a site prior function on torch tensor X_t and return (mean_np, std_np).
    Supports:
      - returns (mean, std) or (mean, var)
      - returns a distribution-like with .mean and .variance
      - returns only mean (std -> None)
    """
    with torch.no_grad():
        out = prior_fn(X_t)

    # tuple (mean, std|var)
    if isinstance(out, tuple) and len(out) >= 2:
        mean_t, second_t = out[:2]
        mean = _to_numpy(mean_t).reshape(-1)
        second = _to_numpy(second_t).reshape(-1)
        # Heuristic: if values look like variance, take sqrt; otherwise assume std
        std = np.sqrt(second) if np.all(second >= 0) else second
        return mean, std

    # distribution-like
    if hasattr(out, "mean"):
        mean_t = out.mean
        var_t = getattr(out, "variance", None)
        mean = _to_numpy(mean_t).reshape(-1)
        if var_t is not None:
            if hasattr(var_t, "to_dense"):
                var_t = var_t.to_dense()
            std = np.sqrt(_to_numpy(var_t).reshape(-1))
        else:
            std = None
        return mean, std

    # tensor-like mean only
    if torch is not None and isinstance(out, torch.Tensor):
        return _to_numpy(out).reshape(-1), None

    # array-like mean only
    arr = np.asarray(out)
    return arr.reshape(-1), None


def plot_y_vs_each_x_from_results_with_site_priors(
    results: Dict[str, Dict[str, object]],          # results[site] = {"X": (n_i,d), "Y": (n_i,)}
    global_gp: object,                               # your trained global GP (used to build site priors)
    global_cache: Dict[str, Dict[str, torch.Tensor]],# global_cache[site]["X"] (torch, holds device/dtype/D)
    feats_norm: Dict[str, object],                   # site -> normalization object used by make_site_prior_fn
    sites_for_prior: Sequence[str],                  # subset of sites to overlay priors for
    *,
    x_labels: Optional[Sequence[str]] = None,
    title_prefix: str = "Warm-start prior",
    figsize: Tuple[float, float] = (7.2, 4.8),
    alpha_points: float = 0.9,
    alpha_band: float = 0.25,
    save_dir: Optional[str] = None,
    fmt: str = "png",
    dpi: int = 160,
    n_grid: int = 200,
    pad_frac: float = 0.03,
    device: Optional["torch.device"] = None,        # if None, pulled from each site's cache tensor
):
    """
    For each X dimension j:
      • scatter Y vs X[:, j] for all sites (colored by site),
      • for each site in `sites_for_prior`, overlay that site's warm-start PRIOR GP mean & 95% CI.
    """
    # --- gather data across all sites for scatter & bounds ---
    site_names, X_chunks, Y_chunks = [], [], []
    d_data = None
    for site, dct in results.items():
        X_i = _to_numpy(dct["X"])
        Y_i = _to_numpy(dct["Y"]).reshape(-1)
        if X_i.ndim != 2:
            raise ValueError(f"results['{site}']['X'] must be 2D, got {X_i.shape}")
        if Y_i.shape[0] != X_i.shape[0]:
            raise ValueError(f"results['{site}']: Y={Y_i.shape[0]} vs X rows={X_i.shape[0]}")
        if d_data is None:
            d_data = X_i.shape[1]
        elif X_i.shape[1] != d_data:
            raise ValueError("All sites must share identical X dimensionality in results.")
        X_chunks.append(X_i); Y_chunks.append(Y_i)
        site_names.extend([site] * X_i.shape[0])

    if d_data is None:
        raise ValueError("Empty results.")

    X_all = np.vstack(X_chunks)        # (N, d_data)
    Y_all = np.concatenate(Y_chunks)   # (N,)
    sites_all = np.array(site_names)

    if not x_labels or len(x_labels) != d_data:
        x_labels = [f"X[{j}]" for j in range(d_data)]

    # axis bounds from pooled data, with a small pad
    mins = X_all.min(axis=0); maxs = X_all.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-9)
    x_bounds = [(mins[j] - pad_frac * spans[j], maxs[j] + pad_frac * spans[j]) for j in range(d_data)]

    # color map per site (backward-compatible)
    uniq_sites = np.unique(sites_all)
    try:
        cmap = plt.cm.get_cmap("tab20", max(len(uniq_sites), 3))
        cmapN = getattr(cmap, "N", max(len(uniq_sites), 3))
    except TypeError:
        cmap = plt.cm.get_cmap("tab20")
        cmapN = getattr(cmap, "N", 20)
    color_map = {s: cmap(i % cmapN) for i, s in enumerate(uniq_sites)}

    # line styles for prior curves (cycle for multiple sites)
    line_styles = ["-", "--", "-.", ":"]
    style_for = {s: line_styles[i % len(line_styles)] for i, s in enumerate(sites_for_prior)}

    # --- plot per dimension ---
    for j in range(d_data):
        fig, ax = plt.subplots(figsize=figsize)

        # scatter by site
        for s in uniq_sites:
            m = (sites_all == s)
            ax.scatter(
                X_all[m, j], Y_all[m],
                label=str(s), s=28, alpha=alpha_points,
                edgecolors="none", c=[color_map[s]], zorder=3
            )

        # overlay warm-start site priors
        first_band_drawn = False
        lo, hi = x_bounds[j]
        xg_np = np.linspace(lo, hi, n_grid)

        for s in sites_for_prior:
            if s not in results or s not in global_cache or s not in feats_norm:
                # Skip silently if a site is missing from any required mapping
                continue

            # dimensions / dtype / device match this site's cache tensor
            X_cache = global_cache[s]["X"]                 # torch.Tensor
            D_site = int(X_cache.size(1))
            if D_site != d_data:
                raise ValueError(
                    f"Site '{s}' prior has D={D_site} but results have d={d_data}. "
                    f"These must match."
                )
            site_device = X_cache.device if device is None else device
            site_dtype  = X_cache.dtype

            # build the prior function
            site_feat_norm = feats_norm[s]
            prior_fn = make_site_prior_fn(global_gp, site_feat_norm, D=D_site, device=site_device)

            # reference vector: median over this site's observed X (from results)
            X_site_np = _to_numpy(results[s]["X"])
            ref_np = np.median(X_site_np, axis=0)   # (D_site,)

            # full grid for this site, varying only dim j
            Xg_t = torch.as_tensor(
                np.tile(ref_np, (n_grid, 1)),
                dtype=site_dtype, device=site_device
            )
            xg_t = torch.as_tensor(xg_np, dtype=site_dtype, device=site_device)
            Xg_t[:, j] = xg_t

            # evaluate prior mean/std & plot
            mu, sd = _eval_prior_fn(prior_fn, Xg_t)   # numpy arrays
            if sd is not None:
                ci_lo = mu - 1.96 * sd
                ci_hi = mu + 1.96 * sd
                ax.fill_between(
                    xg_np, ci_lo, ci_hi,
                    alpha=alpha_band, linewidth=0,
                    color=color_map.get(s, None),
                    label=None if first_band_drawn else "95% CI (prior)",
                    zorder=1
                )
                first_band_drawn = True

            ax.plot(
                xg_np, mu,
                linestyle=style_for[s],
                linewidth=2.0,
                color=color_map.get(s, None),
                label=f"{s} prior mean",
                zorder=2
            )

        # cosmetics
        ax.set_xlabel(x_labels[j]); ax.set_ylabel("Y")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        ax.set_title(f"{title_prefix}: Y vs {x_labels[j]}")

        # legend: sites (scatter) + prior curves (+ one CI entry)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Legend", loc="best",
                  frameon=True, framealpha=0.9, scatterpoints=1, markerscale=1.3)

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            safe = x_labels[j].replace(' ', '_').replace('[','').replace(']','')
            fig.savefig(os.path.join(save_dir, f"y_vs_{safe}.{fmt}"), dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
           
            
# ──────────────────────────────────────────────────────────────────────────────
# 7) MAIN WORKFLOW
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
    
    
    matplotlib.use("Agg")


    #plot_y_vs_each_x_from_results(global_cache, x_labels=None, save_dir="figs/warm_start", fmt="png", dpi=160)
    
    

    sites_subset = ["banfora", "djibo", "po", "reo"]   # <-- your subset
    
    plot_y_vs_each_x_from_results_with_site_priors(
        global_cache,
        global_gp=global_gp,
        global_cache=global_cache,
        feats_norm=feats_norm,
        sites_for_prior=sites_subset,
        x_labels=None,                 # or list of names for X dims
        title_prefix="Warm-start local prior",
        save_dir=None,                 # or "figs/warm_start"
        fmt="png",
        dpi=160,
        n_grid=300
    )


    if not GLOBAL_ONLY:
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
