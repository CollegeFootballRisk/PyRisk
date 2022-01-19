# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 23:02:12 2020

@author: Connor

This file is a sql table compare tool done manually for now.
(i.e. copying the table to a csv for easy parsing)
"""

import numpy as np
from pandas import read_csv
from itertools import product

data1 = read_csv("sql_table_data1.txt", delimiter=" ")
data2 = read_csv("sql_table_data2.txt", delimiter=" ")

data_1 = data1.to_numpy()
data_2 = data2.to_numpy()

# Want to know if 8 or 12 changed specifically.
# The number of people should be the same
# only thing that should change is potentially ID and Team

for ii in range(1, 184):
    inds1 = np.where(data_1[:,11] == ii)[0]
    inds2 = np.where(data_2[:,11] == ii)[0]
    for ind1, ind2 in product(inds1, inds2):
            team1 = data_1[ind1,0]
            if data_2[ind2,0] == team1:
                print("!", end="")
                assert np.all(data_2[ind2, 0:8] == data_1[ind1, 0:8])
                assert data_2[ind2, 9] - data_1[ind1, 9] < 1e-5
                assert np.all(data_2[ind2, 11:13] == data_1[ind1, 11:13])
    
