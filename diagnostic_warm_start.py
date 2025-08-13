#!/usr/bin/env python3
"""
Diagnostic for warm_start.py:
- builds shared Sobol seeds across multiple sites
- fits global GP with product kernel [params] x [site-features]
- checks saved seeds on disk
- constructs a warm prior per site and runs a short Turbo-TS loop
- prints kernel hypers, shapes, and per-site best scores
"""

import os
import json
import time
import numpy as np
import torch

# Import your warm-start module (the one you integrated)
import warm_start as ws


def _print_header(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def main():
    torch.set_default_dtype(torch.double)
    device = "cpu"

    # -------------------------------------------------------------------------
    # 0) Quick environment sanity
    # -------------------------------------------------------------------------
    D = len(ws.param_ranges)
    sites = list(ws.global_sites)
    print(f"Detected D={D} parameters, sites={sites}")
    print(f"Param ranges: {ws.param_ranges}")
    print(f"Using site features for {len(ws.site_features)} sites "
          f"(feature dim={len(next(iter(ws.site_features.values())))})")

    # -------------------------------------------------------------------------
    # 1) GLOBAL SEEDS: evaluate (or load) shared Sobol points on multiple sites
    # -------------------------------------------------------------------------
    _print_header("GLOBAL SEED EVALUATION")
    N_seed = 8  # keep small for a quick diagnostic
    seed_dir = "diag_global_seeds"

    t0 = time.time()
    global_cache = ws.evaluate_global_seeds(
        sites=sites,
        N_seed=N_seed,
        outdir=seed_dir,
        seed=123,
        params={},          # pass through to your simulator as needed
        device=device,
    )
    t1 = time.time()

    # Basic checks and prints
    for s in sites:
        Xs = global_cache[s]["X"]
        Ys = global_cache[s]["Y"]
        print(f"[site {s}] seed X shape={tuple(Xs.shape)} (should be ({N_seed}, {D})) | Y shape={tuple(Ys.shape)}")
        assert Xs.shape == (N_seed, D)
        assert Ys.ndim == 1 and Ys.shape[0] == N_seed
        assert (Xs >= 0).all() and (Xs <= 1).all(), "Seeds must be in [0,1]^D."
        assert torch.isfinite(Ys).all(), "Non-finite Y found."

        # Check files exist
        xpth = os.path.join(seed_dir, f"seed_X_site{s}.pt")
        ypth = os.path.join(seed_dir, f"seed_Y_site{s}.pt")
        assert os.path.exists(xpth) and os.path.exists(ypth), "Seed files missing on disk."

    print(f"Global seed phase took {t1 - t0:.2f}s; files saved in '{seed_dir}'.")

    # Re-load once to ensure idempotency
    _ = ws.evaluate_global_seeds(sites, N_seed, seed_dir, seed=123, params={}, device=device)
    print("Re-loading seeds: OK (files reused).")

    # -------------------------------------------------------------------------
    # 2) GLOBAL GP FIT: product kernel on [params] x [site-features]
    # -------------------------------------------------------------------------
    _print_header("GLOBAL GP FIT (product kernel)")
    t2 = time.time()
    global_gp, feats_norm, (fmin, fmax) = ws.fit_global_gp(
        global_cache=global_cache,
        features=ws.site_features,
        only_sites=sites,
        device=device,
    )
    t3 = time.time()
    print(f"Fitted global GP in {t3 - t2:.2f}s")

    # Introspect kernel hypers (robustly)
    try:
        covar = global_gp.covar_module               # ScaleKernel(ProductKernel(...))
        pk = covar.base_kernel                       # ProductKernel
        k_param, k_site = pk.kernels                 # RBF(params), RBF(site-feats)
        print("Param lengthscale:", k_param.lengthscale.detach().cpu().numpy().ravel())
        print("Site  lengthscale:", k_site.lengthscale.detach().cpu().numpy().ravel())
        print("Output scale:", covar.outputscale.detach().cpu().item())
    except Exception as e:
        print("Kernel hyperparameter introspection failed (non-fatal).", repr(e))

    # -------------------------------------------------------------------------
    # 3) PER-SITE: warm prior + short Turbo Thompson Sampling
    # -------------------------------------------------------------------------
    _print_header("PER-SITE TURBO (warm prior)")
    results = {}
    quick_iters = 6           # small number for a lightweight diagnostic
    batch_size = 2
    n_candidates = 512

    for s in sites:
        # Build the warm prior mean μ_site(x_unit)
        D_unit = global_cache[s]["X"].size(1)
        site_feat_norm = feats_norm[s]
        prior_fn = ws.make_site_prior_fn(global_gp, site_feat_norm, D=D_unit, device=device)

        # Show prior at 2 random unit-cube points
        X_check = torch.rand(2, D_unit, dtype=torch.double, device=device)
        prior_vals = prior_fn(X_check).detach().cpu().numpy()
        print(f"[site {s}] prior μ(x) at 2 random points: {np.round(prior_vals, 6)}")

        # Run a short TR-TS loop
        best_x_unit, best_y, X_hist, Y_hist = ws.run_site_turbo(
            site=s,
            cached=global_cache[s],     # reuse seed data (no re-sim)
            prior_fn=prior_fn,
            n_iter=quick_iters,
            batch_size=batch_size,
            n_candidates=n_candidates,
            device=device,
        )

        # Diagnostics
        initial_best = float(global_cache[s]["Y"].min())
        improved = initial_best - float(best_y)
        print(f"[site {s}] initial best={initial_best:.6f}  ->  final best={best_y:.6f}  (Δ={improved:+.6f})")
        assert torch.isfinite(X_hist).all() and torch.isfinite(Y_hist).all()
        assert (X_hist >= 0).all() and (X_hist <= 1).all(), "History X must remain in [0,1]^D."
        results[s] = {
            "final_best_unitX": best_x_unit.detach().cpu().numpy().tolist(),
            "final_best_score": float(best_y),
            "n_evals": int(Y_hist.shape[0]),
        }

        # Save per-site diagnostics
        os.makedirs("diag_site_results", exist_ok=True)
        torch.save(X_hist.cpu(), f"diag_site_results/X_hist_site{s}.pt")
        torch.save(Y_hist.cpu(), f"diag_site_results/Y_hist_site{s}.pt")

    # -------------------------------------------------------------------------
    # 4) Write summary JSON
    # -------------------------------------------------------------------------
    summary_path = "diag_site_results/summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
