import dataclasses, json
import json
import os
import math
import numpy as np
import pdb
import torch

import gc
from collections import Counter

from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.acquisition.objective import IdentityMCObjective, GenericMCObjective
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from dataclasses import dataclass
#from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import IIDNormalSampler

from torch.quasirandom import SobolEngine

import pathlib

from .batch_generator import BatchGenerator 

import gpytorch

torch.set_default_dtype(torch.float64)


@dataclass
# class TurboThompsonSampling():
class TurboThompsonSampling(BatchGenerator):
    dim: int
    batch_size: int
    length: float = 1.0
    length_min: float =2**-30
    length_max: float = 1.0
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = float("nan")  # Note: Post-initialized # Note: The original paper uses 3
    y_max: float = -float("inf")
    n_candidates: int = None
    dtype = None
    device = None

    def __init__(self, dim=1, batch_size=4, failure_tolerance = None, success_tolerance = None, n_candidates=None, objective=IdentityMCMultiOutputObjective(), dtype=torch.double, device=torch.device("cpu"), length_min=2**-30):
        super().__init__()
        self.dim = dim
        self.batch_size = batch_size
        self.n_candidates = n_candidates
        self.dtype = dtype
        self.device = device
        self.length_min = length_min
        self.objective = objective
        self.length = 1.0

        if failure_tolerance is None:
            self.failure_tolerance = math.ceil(
                max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
            )
        else:
            self.failure_tolerance = failure_tolerance
            
        if self.success_tolerance is None:
            self.success_tolerance = 8
            
        else:
            self.success_tolerance = success_tolerance

        if self.n_candidates is None:
            self.n_candidates = min(5000, max(2000, 200 * self.dim))

    def update(self, X, Y):
        Y = torch.transpose(Y,0,1)
        max_score = torch.max(Y)
        if max_score.item() > self.y_max: # 1e-3
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length /= 2.0
            self.failure_counter = 0
        
        self.y_max = max(self.y_max, max_score.item())
        if self.length < self.length_min:
            self.stopping_condition = True

        return self

    def generate_batch(self, model, X, Y):
        
        print("Generating Batch", flush=True)
        
        X_cand = self._create_candidates(model, X, Y)
        
        #print("Making predictions...",flush=True)
        model.eval()
        model.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(10000):
            predictions = model.likelihood(model(X_cand.to(torch.double)))
            mean = predictions.mean.detach()
            #lower, upper = predictions.confidence_region()
        
        print("Selecting next samples...",flush=True)
        if mean.dim()>1 and mean.shape[-1] > 1:
            mean = mean.sum(dim=1)

        sorted,indices = torch.sort(mean)
            
        o = indices[-self.batch_size:].numpy().tolist()
        X_next = X_cand[o]
        return X_next
   
    def _create_candidates(self, model, X, Y):
        
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

        YO = self.objective(Y)

        self.update(X, YO)
        if self.stopping_condition:
            return

        # Scale the TR to be proportional to the lengthscales
        x_center = X[YO.argmax(), :].clone()
        weights = torch.ones(X.shape[-1])

        try:
            weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        except Exception as e:
            try:
                weights = model.covar_module.data_covar_module.lengthscale.squeeze().detach()
            except Exception as e:
                pass
       
        weights = weights / weights.mean()
        # pdb.set_trace()
        if len(weights.shape) == 0: weights = weights.unsqueeze(-1)

        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * self.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.length / 2.0, 0.0, 1.0)

        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(self.n_candidates).to(dtype=self.dtype, device=self.device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(self.n_candidates, dim, dtype=self.dtype, device=self.device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        if len(ind) > 0:
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

        # Create candidate points from the perturbations and the mask        
        X_cand = x_center.expand(self.n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        return X_cand
      

    def write_checkpoint(self, checkpointdir):
        with open(checkpointdir + "/TurboThompson.json", "w") as statefile:
            json.dump(dataclasses.asdict(self), statefile, ensure_ascii=False, indent=4)
            return True
        return False

    def read_checkpoint(self, checkpointdir):
        with open(checkpointdir + "/TurboThompson.json") as statefile:
            s = json.load(statefile)
            self.dim = s['dim']
            self.batch_size = s['batch_size']
            self.length = s['length']
            self.length_min = s['length_min']
            self.length_max = s['length_max']
            self.failure_counter = s['failure_counter']
            self.failure_tolerance = s['failure_tolerance']
            self.success_counter = s['success_counter']
            self.success_tolerance = s['success_tolerance']
            self.y_max = s['y_max']
            self.n_candidates = s['n_candidates']
            return True
        return False
