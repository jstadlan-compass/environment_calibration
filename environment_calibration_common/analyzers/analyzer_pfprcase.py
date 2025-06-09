##### Import required packages #####
# standard packages
import argparse
import os
from idmtools.analysis.analyze_manager import AnalyzeManager
from idmtools.core import ItemType
from idmtools.core.platform_factory import Platform

# from within analyzers/
from .analyzer_collection import (EventReporterAnalyzer, 
                                MonthlyPfPRAnalyzer,
                                InsetChartAnalyzer)
# from source 'simulations' directory
sys.path.append("../../simulations")
import manifest

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', dest='expt_name', type=str, required=False)
    parser.add_argument('-id', dest='expt_id', type=str, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    simyears = 30
    args = parse_args()
    wdir = os.path.join(manifest.output_dir, args.expt_name)

    if not os.path.exists(wdir):
        os.makedirs(wdir)

    sweep_variables = ['Run_Number', 'Habitat_Type', 'Habitat_Multiplier', 
                       'Constant_Multiplier', 'CM_Coverage']
    channels = ['Adult Vectors', 'Daily EIR', 'Daily Bites per Human', 'Rainfall']

    with Platform('SLURM_LOCAL', job_directory=manifest.job_directory) as platform:
        analyzers = []
        analyzers.append(MonthlyPfPRAnalyzer(sweep_variables=sweep_variables,
                                            start_year=2015,
                                            end_year=2019,
                                            working_dir=wdir))
        analyzers.append(EventReporterAnalyzer(sweep_variables=sweep_variables,
                                               working_dir=wdir))
        analyzers.append(InsetChartAnalyzer(channels=channels,
                                            sweep_variables=sweep_variables,
                                            working_dir=wdir,
                                            start_day=(simyears-4)*365,
                                            end_day=99999))

        manager = AnalyzeManager(configuration={},
                                 ids=[(args.expt_id, ItemType.EXPERIMENT)],
                                 analyzers=analyzers, partial_analyze_ok=True)
        
        manager.analyze()
        print(f"\nSaving outputs to: {wdir}")
