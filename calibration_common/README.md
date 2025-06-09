# calibration_common
Shared files between botorch calibration workflows

### bo.py

Includes functions that control the entire calibration workflow, such as reading checkpoints, creating initial parameter sets, and what should be performed during each step/iteration of the workflow. Typically these can be left as is but note that this is where you may need to add changes for modifying the inner machinery of calibration.

### batch_generators

Includes basic batch generators using the various acquisition functions. Expected_improvement.py and turbo_thompson_sampling.py should be functional for QUEST. Others may require some small improvements related to botorch functionalities.

* generic
* array
* expected_improvement
* thompson_sampling
* turbo_expected_improvement
* **turbo_thompson_sampling**
* turbo_thompson_sampling_local
* turbo_upper_confidence_bound
* upper_confidence_bound


### emulators

GP.py defining the basic setup for the GP Emulation. This script contains additional possible emulators; however, we have only used ExactGP and explored ExactMultiTaskGP so far.

* **Exact GP**
* Exact GP Turbo Local
* Exact GP Fixed Noise
* Exact Multitask GP
* Approximate GPs...
