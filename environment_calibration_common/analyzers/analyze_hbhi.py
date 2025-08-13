##### Import required packages #####
# standard packages
import argparse
import os
import sys
from idmtools.analysis.analyze_manager import AnalyzeManager
from idmtools.core import ItemType
from idmtools.core.platform_factory import Platform

# from within analyzers/
sys.path.append(os.path.dirname(__file__))
from .analyzer_collection import (
    EventReporterAnalyzer,
    MonthlyPfPRAnalyzer,
    MonthlyIncidenceAnalyzer,
    AnnualPfPRAnalyzer,
    AnnualIncidenceAnalyzer,
    InsetChartAnalyzer,
    EventReporterSummaryAnalyzer,
    NodeDemographicsAnalyzer,
    VectorStatsAnalyzer,
    EIRAnalyzer,
    PCRAnalyzer
)
# from within environment_calibration_common submodule
sys.path.append("../")
from helpers import load_coordinator_df

# from source 'simulations' directory
sys.path.append("../../simulations")
import manifest
year_start = 2005
year_end = 2022


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", dest="site", type=str, required=True)
    parser.add_argument("--expid", dest="expid", type=str, required=True)

    return parser.parse_args()


def analyze_experiment(platform, expid, wdir, yearstart, yearend):
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    sim_years = year_end - year_start
    analyzers = []
    # custom analyzers
    sweep_variables = ['Run_Number', 'Sample_ID']
    # EIR analyzer
    analyzers.append(EIRAnalyzer(sweep_variables=sweep_variables,
                                                working_dir=wdir,
                                                start_day=min(0,(sim_years-10)*365),
                                                end_day = sim_years*365,
                                                channels=["Daily EIR"]))



    analyzers.append(MonthlyPfPRAnalyzer(sweep_variables=sweep_variables,
                                         working_dir=wdir,
                                         start_year=year_start,
                                         end_year=year_end))
    analyzers.append(AnnualPfPRAnalyzer(sweep_variables=sweep_variables,
                                        working_dir=wdir,
                                        start_year=year_start,
                                        end_year=year_end))
    analyzers.append(MonthlyIncidenceAnalyzer(sweep_variables=sweep_variables,
                                             working_dir=wdir,
                                             start_year=year_start,
                                             end_year=year_end+1))
    analyzers.append(AnnualIncidenceAnalyzer(sweep_variables=sweep_variables,
                                             working_dir=wdir,
                                             start_year=year_start,
                                             end_year=year_end+1))                                    
    manager = AnalyzeManager(platform=platform,
                             configuration={},
                             ids=[(expid, ItemType.EXPERIMENT)],
                             analyzers=analyzers,
                             partial_analyze_ok=True,
                             max_workers=16)
    
    manager.analyze()


if __name__ == "__main__":
    
    args = parse_args()
    job_directory = "gpfs/home/ykc2461/environments/hbhi_server/malaria-gn-hbhi/IO/experiments_emodpy/ykc2461_simUncert_gin_2005-2022_3x"
    output_dir =  "gpfs/home/ykc2461/simulation_outputs" 
    platform = Platform('SLURM_LOCAL', job_directory=job_directory)
    outdir = args.site
    analyze_experiment(platform, 
                       args.expid,
                       os.path.join(output_dir, outdir))
