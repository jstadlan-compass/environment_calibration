import os, json, csv
from datetime import datetime, timedelta
import pandas as pd

# 1) Precompute your month labels
exp_dir = 'gpfs/projects/b1139/malaria-gn-hbhi/IO/experiments_emodpy/ykc2461_simUncert_gin_2005-2022_3x/ykc2461_simUncert_gin_2005-2022_3x_2025_07_17_16_55/'
start = datetime(2005,1,1)
end   = datetime(2022,12,31)
months = pd.date_range(start=start, end=end, freq='MS')
month_strs = [m.strftime('%Y-%m') for m in months]

# 2) Open three CSVs for streaming
header = ['DS_Name','Sample_ID','Run_Number'] + month_strs
with open('monthly_eir.csv',              'w', newline='') as feir, \
     open('monthly_incidence.csv',   'w', newline='') as fninf, \
     open('monthly_true_prevalence.csv',  'w', newline='') as fp:

    weir  = csv.writer(feir)
    winf  = csv.writer(fninf)
    wprev = csv.writer(fp)
    weir.writerow(header)
    winf.writerow(header)
    wprev.writerow(header)

    # 3) Iterate folders, one at a time
    for sim in os.listdir(exp_dir):
        meta_path   = os.path.join(sim,'metadata.json')
        report_path = os.path.join(sim,'output','ReportMalariaFiltered.json')
        if not os.path.isfile(meta_path) or not os.path.isfile(report_path):
            continue

        # 3a) load metadata
        with open(meta_path) as f: tags = json.load(f).get('tags',{})
        ds   = tags.get('DS_NAME') or tags.get('DS_Name')
        sid  = tags.get('Sample_ID')
        run  = tags.get('Run_Number')

        # 3b) load report
        with open(report_path) as f: rpt = json.load(f)['Channels']
        eir    = rpt['Daily EIR']['Data']
        prev   = rpt['True Prevalence']['Data']
        newinf = rpt['New Infections']['Data']
        days   = len(eir)

        # 3c) build month‑aggregates
        #    (very similar to before but collect into lists)
        sums_eir    = [0]*len(month_strs)
        sums_inf    = [0]*len(month_strs)
        sums_prev   = [0]*len(month_strs)
        counts_prev = [0]*len(month_strs)

        for day in range(days):
            dt = start + timedelta(days=day)
            if dt > end: break
            idx = (dt.year - 2005)*12 + (dt.month-1)
            sums_eir[idx]    += eir[day]
            sums_inf[idx]    += newinf[day]
            sums_prev[idx]   += prev[day]
            counts_prev[idx] += 1

        avgs_prev = [
            (sums_prev[i]/counts_prev[i]) if counts_prev[i]>0 else ''
            for i in range(len(month_strs))
        ]

        # 3d) write one row per metric
        base = [ds, sid, run]
        weir.writerow( base + sums_eir    )
        winf.writerow( base + sums_inf    )
        wprev.writerow(base + avgs_prev   )

        # optional: flush periodically so you never lose more than a chunk
        feir.flush(); fninf.flush(); fp.flush()
