#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 11:40:41 2025

@author: ykc2461
"""

import numpy as np
import pandas as pd

df = pd.read_csv("test.csv")

print(df.head())

    

df_exploded = pd.concat([pd.DataFrame({ col: np.vstack(np.fromstring(row[col].strip('[]'), sep=',')) for col in df.columns[1:]}) for _, row in df.iterrows()], axis=0)

#df_exploded = pd.concat([pd.DataFrame({ col: row[col] for col in df.columns[] }) for _, row in df.iterrows()],, axis=0)