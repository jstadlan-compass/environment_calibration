#!/usr/bin/env python3
"""
Plot global MTGP mean profiles per parameter with sampled points per site.

- Uses warm_start.py functionality:
  * evaluate_global_seeds(...) to get multi-site seed data (or load from disk)
  * fit_global_gp(...) to train the global product-kernel GP
  * make_site_prior_fn(...) to build per-site mean functions μ_site(x_unit)

Outputs one PNG per parameter in ./gp_param_plots/.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import warm_start as ws


def build_baseline_X(global_cache: dict) -> torch.Tensor:
    """Baseline (unit-cube) parameter vector: mean of all seed X across all sites."""
    all_X = []
    for s in global_cache:
        all_X.append(global_cache[s]["X"])
    X_stack = torch.cat(all_X, dim=0)  # (N_total, D)
    return X_stack.mean(dim=0, keepdim=True)  # (1, D)


def unit_to_real(x_unit: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    return lb + (ub - lb) * x_unit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_dir", type=str, default="global_seeds",
                        help="Directory to read/write shared Sobol seeds per site.")
    parser.add_argument("--n_seed", type=int, default=16,
                        help="Number of shared Sobol seed points per site (if not already saved).")
    parser.add_argument("--sites", type=str, default="", help="Comma-separated site IDs. "
                        "Defaults to warm_start.global_sites if empty.")
    parser.add_argument("--grid", type=int, default=200, help="Grid points per profile curve.")
    parser.add_argument("--outdir", type=str, default="gp_param_plots",
                        help="Directory to save plots.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.set_default_dtype(torch.double)
    device = args.device

    # Resolve sites
    if args.sites.strip():
        sites = [s.strip() for s in args.sites.split(",")]
        # If warm_start used ints, coerce
        sites = [int(s) if s.isdigit() else s for s in sites]
    else:
        sites = list(ws.global_sites)

    # Evaluate (or load) global multi-site Sobol seeds
    global_cache = ws.evaluate_global_seeds(
        sites=sites,
        N_seed=args.n_seed,
        outdir=args.seed_dir,
        seed=123,
        params={},        # pass-through to your simulator if needed
        device=device
    )

    # Train global GP (product kernel over [X_unit] × [site_features_norm])
    global_gp, feats_norm, _ = ws.fit_global_gp(
        global_cache=global_cache,
        features=ws.site_features,
        only_sites=sites,
        device=device
    )

    # Prep param bounds and names
    lb, ub, names = ws.ranges_to_tensors(ws.param_ranges, device=device, dtype=torch.double)
    D = len(names)

    # Baseline unit-cube vector (others fixed while sweeping one parameter)
    x_base = build_baseline_X(global_cache).to(device=device)  # (1, D)

    # Colors per site
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"])
    site_colors = {s: color_cycle[i % len(color_cycle)] for i, s in enumerate(sites)}

    os.makedirs(args.outdir, exist_ok=True)

    for d, pname in enumerate(names):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))

        # Scatter: sampled seed points for each site (X[:, d] vs Y)
        for s in sites:
            Xs_unit = global_cache[s]["X"].to(device=device)       # (N_seed, D)
            Ys      = global_cache[s]["Y"].to(device=device)       # (N_seed,)
            xd_real = unit_to_real(Xs_unit[:, d], lb[d], ub[d]).cpu().numpy()
            ax.scatter(
                xd_real, Ys.cpu().numpy(),
                s=24, alpha=0.8, label=f"site {s} (samples)",
                color=site_colors[s], edgecolors="none"
            )

        # Mean curves per site along parameter d (others fixed at baseline)
        x_grid_unit = torch.linspace(0.0, 1.0, args.grid, dtype=torch.double, device=device)
        x_grid_real = unit_to_real(x_grid_unit, lb[d], ub[d]).cpu().numpy()

        # Shared base replicated, then overwrite column d
        Xgrid = x_base.repeat(args.grid, 1)  # (G, D)
        Xgrid[:, d] = x_grid_unit

        for s in sites:
            prior_fn = ws.make_site_prior_fn(
                global_gp=global_gp,
                site_feature_norm=feats_norm[s],
                D=D,
                device=device
            )
            with torch.no_grad():
                y_mean = prior_fn(Xgrid).cpu().numpy()
            ax.plot(
                x_grid_real, y_mean,
                lw=2.0, color=site_colors[s],
                label=f"site {s} (GP mean)"
            )

        ax.set_xlabel(f"{pname} (real units)")
        ax.set_ylabel("Calibration score (GP mean / samples)")
        ax.set_title(f"Global MTGP profiles — parameter: {pname}")
        ax.grid(True, alpha=0.25)
        # Build a clean legend: combine unique labels
        handles, labels = ax.get_legend_handles_labels()
        # de-duplicate legend entries (site twice for samples+mean)
        uniq = {}
        for h, l in zip(handles, labels):
            uniq[l] = h
        ax.legend(uniq.values(), uniq.keys(), fontsize=9, ncol=2)

        outpath = os.path.join(args.outdir, f"global_gp_profile_param{d+1}_{pname}.png")
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
