import os
import pandas as pd
import numpy as np
import torch
from my_func import my_func_per_site as myFunc
from translate_parameters import translate_parameters

import manifest as manifest

def run_simulation_for_site(site, exp_label, output_dir, calib_coord_path, param_key_path, gs_coord_path, weights_path):
    """
    Runs seed-point simulations and scoring for a given site, using its row in global_simulation_coordinator.csv.

    Args:
        site (str): Site name to look up in global_simulation_coordinator.csv
        exp_label (str): Experiment label
        output_dir (str): Output directory
        calib_coord_path (str): Path to calibration coordinator CSV
        param_key_path (str): Path to parameter_key.csv
        gs_coord_path (str): Path to global_simulation_coordinator.csv
        weights_path (str): Path to weights.csv
    """

    # Load calibration coordinator info for this site
    calib_coord = pd.read_csv(calib_coord_path).set_index("site")
    init_samples = int(calib_coord.at[site, "init_size"])

    # Load parameter key
    param_key = pd.read_csv(param_key_path)

    # Load all coordinator sets
    gs_coord_df = pd.read_csv(gs_coord_path)

    # Find the row for the specified site
    site_row = gs_coord_df.loc[gs_coord_df["site"] == site]
    if site_row.empty:
        raise ValueError(f"Site '{site}' not found in {gs_coord_path}")

    # Convert the row to a coord_df-like DataFrame: index is keys, column is 'value'
    row_dict = site_row.iloc[0].to_dict()
    coord_instance = pd.DataFrame(list(row_dict.items()), columns=["option", "value"]).set_index("option")


    # Generate random seed points
    dim = param_key.shape[0]
    seed_points = np.random.rand(init_samples, dim)
    X = torch.tensor(seed_points, dtype=torch.float32)

    # Run simulation for this site
    wdir = os.path.join(output_dir, f"LF_{site}")
    os.makedirs(wdir, exist_ok=True)
    Y0 = myFunc(X, coord_instance)

    # Clean up non-score columns and calculate total score
    ps = Y0["param_set"]
    Y0_scores = Y0.filter(like="_score")
    Y0_scores["param_set"] = ps

    # Weighted sum of scores (see weights.csv for weights)
    weights = pd.read_csv(weights_path).set_index("score_type")["weight"].to_dict()
    def weighted_score(row):
        score = 0
        for col in Y0_scores.columns:
            if col.endswith("_score"):
                wt = weights.get(col.replace("_score", ""), 1)
                val = row[col] if pd.notnull(row[col]) else 10  # missing/NA penalty
                score += wt * val
        return -score  # Negate for maximization

    Y0_scores["total_score"] = Y0_scores.apply(weighted_score, axis=1)

    print(f"Simulation results for {site}:")
    print(Y0_scores[["param_set", "total_score"]])

    Y0_scores.to_csv(os.path.join(wdir, "seed_points_scores.csv"), index=False)
    
    return Y0_scores

# Example usage:fr
if __name__ == "__main__":
    # You can use manifest for these arguments, or specify them directly
    Site = manifest.SITE
    exp_label = manifest.EXPERIMENT_LABEL
    output_dir = f"output/{exp_label}"
    calib_coord_path = manifest.calibration_coordinator_path
    param_key_path = manifest.parameter_key_path
    gs_coord_path = manifest.global_simulation_coordinator_path
    weights_path = "simulation_inputs/weights.csv"

    run_simulation_for_site(Site, exp_label, output_dir, calib_coord_path, param_key_path, gs_coord_path, weights_path)
