import os
import pandas as pd
import numpy as np
import torch
from helpers import load_coordinator_df
from my_func import my_func as myFunc
from translate_parameters import translate_parameters

# Load experiment and calibration parameters
import manifest as manifest

# Site and experiment label (edit these for your run)
Site = manifest.SITE
exp_label = manifest.EXPERIMENT_LABEL

output_dir = f"output/{exp_label}"
os.makedirs(output_dir, exist_ok=True)

# Load calibration coordinator info for this site
calib_coord = pd.read_csv(manifest.calibration_coordinator_path).set_index("site")
init_samples = int(calib_coord.at[Site, "init_size"])
init_batches = int(calib_coord.at[Site, "init_batches"])

# Load parameter key and coordinator definition
param_key = pd.read_csv("parameter_key.csv")
coord_df = load_coordinator_df()
incidence_agebin = float(coord_df.at["incidence_comparison_agebin", "value"])
prevalence_agebin = float(coord_df.at["prevalence_comparison_agebin", "value"])

# Generate random seed points in [0, 1]^dim
dim = param_key.shape[0]
seed_points = np.random.rand(init_samples, dim)
X = torch.tensor(seed_points, dtype=torch.float32)

# Call the simulation function on your seed points
wdir = os.path.join(output_dir, "LF_0")
os.makedirs(wdir, exist_ok=True)
Y0 = myFunc(X, wdir)

# Clean up non-score columns and calculate total score
ps = Y0["param_set"]
Y0_scores = Y0.filter(like="_score")
Y0_scores["param_set"] = ps

# Weighted sum of scores (see weights.csv for weights)
weights = pd.read_csv("simulation_inputs/weights.csv").set_index("score_type")["weight"].to_dict()
def weighted_score(row):
    score = 0
    for col in Y0_scores.columns:
        if col.endswith("_score"):
            wt = weights.get(col.replace("_score", ""), 1)
            val = row[col] if pd.notnull(row[col]) else 10  # missing/NA penalty
            score += wt * val
    return -score  # Negate for maximization

Y0_scores["total_score"] = Y0_scores.apply(weighted_score, axis=1)

# Output seed points and their scores
print("Seed points (unit cube):")
print(X)
print("\nSimulation scores:")
print(Y0_scores[["param_set", "total_score"]])

# Save results for later analysis
Y0_scores.to_csv(os.path.join(wdir, "seed_points_scores.csv"), index=False)