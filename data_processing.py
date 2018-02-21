'''
Date: 02/20/2018
Objective:
  - read in ambient bus stop in/out data from raw csv
  - get realtime occupancy on discrete timestep (didn't consider the time for people to enter building)
'''
import csv
import numpy as np
import timestring
from os import listdir
import dateparser
import time
from dateutil import parser


# get in/out bound dict
def getDict(filename):
    boundDict = {}
    with open(filename) as f:
        content = f.readlines()
        for line in content:
            line = line.strip().split(',')
            boundDict[line[0]] = line[1:]
    return boundDict



'''
read in ambient bus stop
return a list of list with components:
  - busID: bus number + bound indicator
  - stopID: name of bus stop considered
  - date: day/month/year
  - arriving time: H-m-s
  - leaving  time: H-m-s
  - output occupancy: number of people exist
  - input  occupancy: number of people enter
'''
def read_csv(filename, inBoundDict, outBoundDict):
    with open(filename) as f:
        content = csv.reader(f)
        infoList = []
        for i, row in enumerate(content):
            tempList = [0]*7
            busID = row[2] + '-' + row[1]
            stopID = row[9]
            # check if in the ambient
            if inBoundDict.get(busID) != None or outBoundDict.get(busID) != None:
                if stopID in inBoundDict.get(busID) or stopID in outBoundDict.get(busID):
                    tempList[0] = busID
                    tempList[1] = stopID
                    tempList[2] = row[6]
                    tempList[3] = row[10] + '-' + row[11] + '-' + row[12]
                    tempList[4] = row[13] + '-' + row[14] + '-' + row[15]
                    tempList[5] = row[16]
                    tempList[6] = row[17]
                    infoList.append(tempList)
    return infoList



def find_current_second(dateStr):
    a = dateparser.parse(str(dateStr)).timetuple()
    a = time.mktime(a)
    return a


# get seconds if over 24
def get_seconds(dateStr, timeStr):
    const = 86400.0
    currentSec = 0.0
    # corner case: 24, 25 in time to 0, 1 of next day
    if timeStr.split(':')[0] == '24' or timeStr.split(':')[0] == '25':
        tempSec = (float(timeStr.split(':')[0])-24.0)*3600.0 + float(timeStr.split(':')[1])*60.0 + float(timeStr.split(':')[2])
        currentSec = find_current_second(dateStr) + const + tempSec
    else:
        tempStr = dateStr + ' ' + timeStr
        currentSec = find_current_second(tempStr)
    return currentSec


'''
get daily occupancy trend from trans
Input:
    - inputArr: read from csv file (list of list)
    - startDate: YYYY-MM-DD
    - endDate: YYYY-MM-DD
    - dT: time interval, in minute
Output:
    - trendArr: n*m array, where n is num of day, m is number of observation
Notice:
    - 86400.0 number of second in one day
    - need to considered the initial value of the first day
'''
def getDailyTrend(inputArr, startDate, endDate, dT):
    # get number of day
    const = 86400.0
    startSec = find_current_second(startDate)
    endSec = find_current_second(endDate)
    deltaSec = dT*60.0          # in second
    numOfDay = int(np.ceil((endSec - startSec)/const))
    numOfObs = int(24*60/dT)
    trendArr = np.zeros((numOfDay, numOfObs))
    idx = 1
    for line in inputArr:
        print idx
        idx += 1
        # convert format
        tempDate = line[2]
        tempArriveTime = line[3].replace('-', ':')
        tempLeaveTime = line[4].replace('-', ':')
        tempOcc = int(line[6]) - int(line[5])
        # convert str time to second
        # corner case: 24, 25 in time to 0, 1 of next day (only account for arrive time)
        tempSec = get_seconds(tempDate, tempArriveTime)
        # find index of day
        i = np.floor((tempSec - startSec)/const).astype(int)
        # find slot of the day
        j = np.floor(((tempSec - startSec)%const)/deltaSec).astype(int)
        trendArr[i,j] += tempOcc
    return trendArr


'''
density to trend
'''
def densityToTrend(inputArr):
    m, n = inputArr.shape
    outputArr = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            outputArr[i, j] = np.sum(inputArr[i, 0:j]) + inputArr[i, j]
    return outputArr


# test
inBoundFile = '/Users/lixuan/Desktop/trans/in bound.txt'
outBoundFile = '/Users/lixuan/Desktop/trans/in bound.txt'
inBoundDict = getDict(inBoundFile)
outBoundDict = getDict(outBoundFile)


filename = '/Users/lixuan/Desktop/trans/1706.csv'

#dataArr = read_csv(filename, inBoundDict, outBoundDict)

#np.save('result.npy', dataArr)


dataArr = np.load('result.npy')

print 'process 1 finished'

'''
csv = open('test.csv', "w")
for line in d:
    print line
    row = ','.join(line)
    csv.write(row)
    csv.write('\n')
'''


startDate = '2017-06-01 00:00:00'
endDate = '2017-07-02 00:00:00'
dT = 5.0

result = getDailyTrend(dataArr, startDate, endDate, dT)

print result[20]

result = densityToTrend(result)


print result[20]
