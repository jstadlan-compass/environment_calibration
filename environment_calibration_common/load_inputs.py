##### Import required packages #####
# from within environment_calibration_common submodule
from helpers import load_coordinator_df


def load_sites():
    coord_df = load_coordinator_df(characteristic=False, set_index=True)
    sites = coord_df.index.tolist()
    nSims = coord_df['nSims'].tolist()
    return sites, nSims#, script_names


if __name__ == '__main__':
    load_sites()
