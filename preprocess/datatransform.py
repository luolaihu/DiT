#!/usr/bin/python3
# -*- coding:utf8 -*-

import os
import datetime
import json
import collections

# Global Dict
CLUSTERMAP = collections.defaultdict(int)
WEATHERMAP = dict()
POIMAP = dict()
TRAFFICMAP = dict()
FEATURESET = set()
FEATUREDICT = dict()

def dateSlice(date):
    sliceTime= datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    day = date[0: date.find(' ')]
    baseTime= datetime.datetime.strptime(day, '%Y-%m-%d')
    minute = (sliceTime - baseTime).seconds / 60
    slice = int(minute) // 10 + 1
    return day + '-' + str(slice)

def loadOrder(dirPath, baseDir):
    global CLUSTERMAP
    global LABELMAP

    files = os.listdir(dirPath)
    for name in files:
        if name.startswith('.'):
            continue
        fullName = os.path.join(dirPath, name)
        reqMap = collections.defaultdict(float)
        ansMap = collections.defaultdict(float)
        with open(fullName, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                if len(items) != 7:
                    print('order format error : %s' % line)
                    continue
                clusterHash = items[3]
                clusterId = CLUSTERMAP[clusterHash]
                if clusterId < 1:
                    print('clusterHash error : %s' % line)
                    continue
                slice = dateSlice(items[6])
                key = str(clusterId) + '#' + slice
                reqMap[key] = reqMap[key] + 1
                if items[1] != 'NULL':
                    ansMap[key] = ansMap[key] + 1

        labelmap = collections.defaultdict(float)
        for k, v in reqMap.items():
            gap = v - ansMap[k]
            labelmap[k] = gap
        with open(os.path.join(baseDir, 'feature'), 'a') as out:
            for key, label in labelmap.items():
                sample = getFeature(key, FEATUREDICT)
                label = str(label)
                out.write(key + '\001' + label + '\001' + sample + '\r\n')
                # feMap['label'] = label
                # out.writelines(json.dumps(feMap))
        print('feature extractor finish : %s' % name)
        reqMap.clear()
        ansMap.clear()
    print('load order finish!')

def loadPoi(dirPath):
    global CLUSTERMAP
    global POIMAP
    global FEATURESET
    files = os.listdir(dirPath)
    for name in files:
        if name.startswith('.'):
            continue
        fullName = os.path.join(dirPath, name)
        with open(fullName, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                clusterHash = items[0]
                clusterId = CLUSTERMAP[clusterHash]
                if clusterId < 1:
                    print('clusterHash error : %s' % line)
                    continue
                pf = dict()
                for i in range(1, len(items)):
                    item = items[i]
                    kv = item.split(':')
                    fe = 'p' + kv[0]
                    pf[fe] = kv[1]
                    FEATURESET.add(fe)
                POIMAP[clusterId] = pf
    print('load poi finish!')

def loadTraffic(dirPath):
    global CLUSTERMAP
    global TRAFFICMAP
    global FEATURESET
    files = os.listdir(dirPath)
    for name in files:
        if name.startswith('.'):
            continue
        fullName = os.path.join(dirPath, name)
        with open(fullName, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                slice = dateSlice(items[len(items) - 1])
                clusterHash = items[0]
                clusterId = CLUSTERMAP[clusterHash]
                if clusterId < 1:
                    print('clusterHash error : %s' % line)
                    continue
                tf = dict()
                for i in range(1, len(items) - 1):
                    item = items[i]
                    kv = item.split(':')
                    fe = 't' + kv[0]
                    tf[fe] = kv[1]
                    FEATURESET.add(fe)
                key = str(clusterId) + '#' + slice
                TRAFFICMAP[key] = tf
    print('load traffic finish!')

def loadWeather(dirPath):
    global CLUSTERMAP
    global WEATHERMAP
    global FEATURESET
    files = os.listdir(dirPath)
    for name in files:
        if name.startswith('.'):
            continue
        fullName = os.path.join(dirPath, name)
        with open(fullName, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                if len(items) != 4:
                    print('weather format error : %s' % line)
                    continue
                slice = dateSlice(items[0])
                wf = dict()
                for i in range(1,4):
                    fe = 'w' + str(i)
                    wf[fe] = items[i]
                    FEATURESET.add(fe)
                WEATHERMAP[slice] = wf
    print('load weather finish!')

def loadClusterMap(dirPath):
    global CLUSTERMAP
    files = os.listdir(dirPath)
    for name in files:
        if name.startswith('.'):
            continue
        fullName = os.path.join(dirPath, name)
        with open(fullName, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                CLUSTERMAP[items[0]] = int(items[1])
    print('load cluster finish!')

def getFeature(key, featureDict):
    feMap = dict()
    if key in TRAFFICMAP:
        tf = TRAFFICMAP[key]
        feMap.update(tf)
    items = key.split('#')
    clusterId = items[0]
    if int(clusterId) in POIMAP:
        pf = POIMAP[int(clusterId)]
        feMap.update(pf)
    dateslice = items[1]
    if dateslice in WEATHERMAP:
        wf = WEATHERMAP[dateslice]
        feMap.update(wf)
    feMap['clid'] = clusterId
    feMap['tid'] = dateslice.split('-')[3]

    sample = ''
    for i in range(1, len(featureDict)):
        if featureDict[str(i)] in feMap:
            wt = feMap[featureDict[str(i)]]
            sample = sample + str(i) + ':' + str(wt) + '\t'
    return sample.strip()

def mapInit(baseDir):
    loadClusterMap(os.path.join(baseDir, 'cluster_map'))
    loadWeather(os.path.join(baseDir, 'weather_data'))
    loadPoi(os.path.join(baseDir, 'poi_data'))
    loadTraffic(os.path.join(baseDir, 'traffic_data'))

def runTrainPipeline():
    baseDir = '/home/luolaihu/Downloads/season_1/training_data'
    mapInit(baseDir)
    print('map init finish!')
    # feature code
    FEATURESET.add('clid')
    FEATURESET.add('tid')
    flist = list(FEATURESET)
    i = 1
    for f in sorted(flist):
        FEATUREDICT[str(i)] = f
        i = i + 1

    with open(os.path.join('/home/luolaihu/Downloads/train/', 'featuredict'), 'a') as fdout:
        fdout.writelines(json.dumps(FEATUREDICT))
    # feature code
    loadOrder(os.path.join(baseDir, 'order_data'), '/home/luolaihu/Downloads/train/')

def runTestPipeline():
    global FEATUREDICT
    baseDir = '/home/luolaihu/Downloads/season_1/test_set_1'
    mapInit(baseDir)
    with open('/home/luolaihu/Downloads/train/featuredict' , 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                FEATUREDICT = json.loads(line)
    loadOrder(os.path.join(baseDir, 'order_data'), '/home/luolaihu/Downloads/test')

if __name__ == '__main__':
    # print(dateSlice('2016-05-24 23:59:00'))
    runTrainPipeline()
    runTestPipeline()
