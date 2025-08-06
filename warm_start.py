import torch
import numpy as np

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import Mean, ConstantMean

from turbo_thompson_sampling import TurboThompsonSampling  # :contentReference[oaicite:4]{index=4}

# -------------------------------------------------------------------
# Your simulator wrapper: 
#   X: NumPy array (D,), params: dict, site: int
# Returns a non-negative float to minimize.
def f_sim(X: np.ndarray, params: dict, site: int) -> float:
    raise NotImplementedError
# -------------------------------------------------------------------

# 1) Problem setup
D      = 4                             # # input dims (habitat multipliers)
sites  = [0,1,2]                       # site IDs
params = {}                            # extra sim parameters

# Each site has its own feature vector (e.g. [lat, lon, pop_density,...])
# shape: (#sites, F)
site_features = {
    0: np.array([0.1, 0.2, 0.3]),
    1: np.array([0.4, 0.5, 0.6]),
    2: np.array([0.7, 0.8, 0.9]),
}
F = len(next(iter(site_features.values())))

# 2) Build global “seed” design & responses
N_seed = 10
seed_X = torch.rand(N_seed, D)
# We'll assemble N_seed*#sites rows of augmented inputs
X_aug_list, Y_aug_list = [], []
for j in range(N_seed):
    xj = seed_X[j]
    for site in sites:
        # simulator knows the true site_features internally if needed
        y = f_sim(xj.numpy(), params, site)
        # augment with site_features (continuous cross-site kernel)
        sf = torch.from_numpy(site_features[site]).float()
        X_aug_list.append(torch.cat([xj, sf]))  # (D+F,)
        Y_aug_list.append(y)

train_X_global = torch.stack(X_aug_list)       # (N_seed*#sites, D+F)
train_Y_global = torch.tensor(Y_aug_list).unsqueeze(-1).double()  # (…,1)

# 3) Fit the global GP with cross-site kernel on site_features
kernel_global = ScaleKernel(RBFKernel(ard_num_dims=D+F))
global_gp = SingleTaskGP(
    train_X=train_X_global,
    train_Y=train_Y_global,
    covar_module=kernel_global,
    mean_module=ConstantMean()
)
mll = ExactMarginalLogLikelihood(global_gp.likelihood, global_gp)
fit_gpytorch_model(mll)
global_gp.eval()

# 4) Build per-site prior-mean functions μ_site(X)
def make_site_prior(global_gp, site):
    sf = torch.from_numpy(site_features[site]).float()
    def μ(x_query: torch.Tensor):
        # x_query: B×D
        B = x_query.size(0)
        sf_rep = sf.unsqueeze(0).repeat(B, 1)       # B×F
        Xq_aug = torch.cat([x_query, sf_rep], dim=-1)  # B×(D+F)
        post = global_gp.posterior(Xq_aug)
        return post.mean.squeeze(-1)               # (B,)
    return μ

# 5) Wrap into a BoTorch Mean module
class PriorMean(Mean):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x)   # x: B×D → (B,)

# 6) Per-site trust-region BO with Turbo Thompson Sampling
results = {}
for site in sites:
    prior_fn = make_site_prior(global_gp, site)
    pm       = PriorMean(prior_fn)

    # (a) initial random observations
    N_init = 5
    X_obs  = torch.rand(N_init, D)
    Y_obs  = torch.tensor(
        [[f_sim(x.numpy(), params, site)] for x in X_obs],
        dtype=torch.double
    )

    # (b) create the TR-TS generator
    tts = TurboThompsonSampling(
        dim=D,
        batch_size=4,
        failure_tolerance=5,
        success_tolerance=8,
        n_candidates=2000
    )

    # (c) trust-region TS loop
    for _ in range(20):
        # fit a warm-start GP surrogate
        covar = ScaleKernel(RBFKernel(ard_num_dims=D))
        gp = SingleTaskGP(
            train_X=X_obs, 
            train_Y=Y_obs,
            covar_module=covar,
            mean_module=pm
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        gp.eval()

        # propose next batch within the current trust region
        X_next = tts.generate_batch(gp, X_obs, Y_obs)   # :contentReference[oaicite:5]{index=5}

        # evaluate simulator
        Y_next = torch.tensor(
            [[f_sim(x.numpy(), params, site)] for x in X_next],
            dtype=torch.double
        )

        # update data & TR state
        X_obs = torch.cat([X_obs, X_next], dim=0)
        Y_obs = torch.cat([Y_obs, Y_next], dim=0)
        tts.update(X_obs, Y_obs)                         # :contentReference[oaicite:6]{index=6}

        if getattr(tts, "stopping_condition", False):
            break

    # record best solution for this site
    best_idx = Y_obs.argmin().item()
    results[site] = (X_obs[best_idx].numpy(), float(Y_obs[best_idx]))

print("Optimal settings per site:", results)
