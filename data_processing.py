'''
Date: 02/20/2018
Objective:
  - read in ambient bus stop in/out data from raw csv
  - get realtime occupancy on discrete timestep (didn't consider the time for people to enter building)
'''
import csv
from os import listdir


# read in ambient bus stop
def read_csv(filename):
    with open(filename) as f:
        content = csv.reader(f)
        infoList = []
        for row in content:
            tempList = []
            #
    return 0




filename = 'C:\\Users\\kkkac\\Desktop\\trans samples\\1706.csv'

d = read_csv(filename)



# out-bound dict
outBoundDict = {}
outBoundDict


# in-bound dict
