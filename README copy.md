# environment_calibration_common

common files 'under the hood' for environment calibration workflow. 

## Repository Contents

#### Some helper files:

* clean_all.py 
* get_eradication.py - Downloads new eradication if manifest doesn't dictate to use a local copy
* get_version.py - Gives version of eradication being used if needed
* load_inputs.py - Loads the coordinator and sites requested
* my_func.py
* run_analyzers.py - Dictates which analyzers to run for which sites based on simulation coordinator. Makes finished.txt when all analyzers for a site are done running.
* run_sims.py - Simulation submision script that manages bringing all of the helpers together to create simulations. During simulation submission we also submit a scheduled waiting script and analyzer script. 
* translate_parameters.py - script for converting locations in unit parameter space to emod-compatible values, and vice-versa, according to 'parameter_key.csv'
* utils_slurm.py - slurm-specific helper functions for chain job submission on QUEST
* helpers.py (modified per IIVT) - Contains general helper functions for simulation setup. If trying to change the core workings of sims, config files, etc, this is the place to do it.
  
#### analyzers

* analyzer_collection.py
* analyze.py

#### compare_to_data

* calculate_all_scores.py
* run_full_comparison.py


## Tips for adding new objectives

To add a new objective (example: vector_species_mix), you may need to

1. Make changes to files in your environment_calibration project directory to allow for control of new objective
    - simulation_inputs/simulation_coordinator.csv :
        - Add logical flag for inclusion (ex. 'vector_mix_comparison')
        - Add reference_dataset path (ex. 'vector_mix_reference')
    - simulation_inputs/weights.csv:
        - Add row with weight for new objective (ex. 'vector_mix_score')
    
    Different objectives may require more controls (ex. agebin, frequency, diagnostic, etc.)
        
2. Make changes to files *here* in the environment_calibration_common module
    - helpers.py : add logic for including EMOD reports in simulation
    - analyzers/analyzer_collection.py : define new analyzer collect simulation output from reporter  
    - analyzers/analyze.py: add logic for when to require analyzer
    - compare_to_data/calculate_all_scores.py :
        - add function to compare outputs of analyzer to reference_data and produce a score by parameter set (ex. 'compare_vector_mix()')
        - add logic to include scores produced in compute_all_scores()
    - compare_to_data/run_full_comparison.py : 
        - add logic to compute_scores_across_site() for weighting score and handling missing values

