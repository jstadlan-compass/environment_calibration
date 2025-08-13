##### Import required packages #####
# standard packages
import os
import re
import sys
# from source 'simulations' directory
sys.path.append("../simulations")
import manifest as manifest

def clean_analyzers():
    for file in os.listdir(manifest.CURRENT_DIR):
        if re.match("run_analyzer_.*sh", file) or re.match("wait_analyzer_.*sh", file):
            os.remove(os.path.join(manifest.CURRENT_DIR,file))

def clean_logs():
    log_dir= manifest.CURRENT_DIR / "log"
    for file in os.listdir(log_dir):
        os.remove(os.path.join(log_dir,file))

def clean_COMPS_ID():
    COMPS_ID_dir = manifest.CURRENT_DIR / "COMPS_ID"
    for file in os.listdir(COMPS_ID_dir):
        if 'version.txt' in file:
            continue
        os.remove(os.path.join(COMPS_ID_dir,file))      


if __name__ == "__main__":
    clean_analyzers()
    clean_logs()
    clean_COMPS_ID()
