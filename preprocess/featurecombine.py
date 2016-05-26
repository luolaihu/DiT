#!/usr/bin/python3
# -*- coding:utf8 -*-

import datetime
import os
import json

COMBINEFE = 3

def featureE(baseFe, cFe):
    CFEATUREMAP = dict()
    LFEATUREMAP = dict()
    DFEATUREMAP = dict()
    with open(baseFe, 'r') as f:
        for line in f:
            items = line.strip().split('\001')
            if len(items) == 3:
                CFEATUREMAP[items[0]] = items[2]
                LFEATUREMAP[items[0]] = items[1].split(',')[0]
                DFEATUREMAP[items[0]] = items[1].split(',')[1]
            else:
                print('format error : %' % line)
    print("load feature finish CFEATUREMAP : %d " % len(CFEATUREMAP))
    print("load feature finish LFEATUREMAP : %d " % len(LFEATUREMAP))
    if os.path.exists(cFe):
        os.remove(cFe)
    with open(cFe, 'a') as out:
        for key, value in CFEATUREMAP.items():
            feStr = ''
            labelStr = ''
            destStr = ''
            label = ''
            if key in LFEATUREMAP:
                label = LFEATUREMAP[key]
            else:
                print('no label:%s' % key)
                continue
            for i in range(0, COMBINEFE):
                items = key.split('-')
                if len(items) == 4:
                    minute = int(items[3])
                    day = items[2]
                    month = items[1]
                    year = items[0].split('#')[1]
                    clid = items[0].split('#')[0]
                    minute = minute - i
                    dateStr = year + '-' + month + '-' + day
                    sliceTime = datetime.datetime.strptime(dateStr, '%Y-%m-%d')
                    deltaMinute = datetime.timedelta(minutes= i * -10)
                    if minute <= 0:
                        minute = minute + 144
                        dateStr = datetime.datetime.strftime(sliceTime + deltaMinute, '%Y-%m-%d')
                    ikey = clid + '#' + dateStr + '-' + str(minute)
                    # print('ikey %s : key %s' % (ikey, key) )
                    if ikey in CFEATUREMAP:
                        fe = CFEATUREMAP[ikey]
                        fitems = fe.split('\t')
                        for fitem in fitems:
                            kvs = fitem.split(':')
                            fid = int(kvs[0])
                            fid = fid + i * 185
                            nkvs = str(fid) + ':' + kvs[1]
                            feStr = feStr + nkvs + '\t'
                    # print('key %s : lastStr %s' % (ikey, feStr))

                    if ikey in LFEATUREMAP:
                        k = COMBINEFE * 185 + i
                        fe = LFEATUREMAP[ikey]
                        if i == 0 :
                            fe = '0'
                        labelStr = labelStr + str(k) + ':' + fe + '\t'

                    if ikey in DFEATUREMAP:
                        k = COMBINEFE * 186 + i
                        fe = DFEATUREMAP[ikey]
                        if i == 0:
                            fe = '0'
                        destStr = destStr + str(k) + ':' + fe + '\t'
                    # print('key %s : lastStr %s' % (ikey , lastStr))
            if len(label.strip()) > 0 :
                out.write(label + '\t' + feStr.strip() + '\t' + labelStr.strip() + '\t' + destStr.strip() + '\r\n')
    del CFEATUREMAP
    del LFEATUREMAP
    del DFEATUREMAP

def trainFe():
    baseFe = '/home/luolaihu/Downloads/train/feature'
    cFe = os.path.join('/home/luolaihu/Downloads/train', 'cfeature')
    featureE(baseFe, cFe)

def evalFe():
    baseFe = '/home/luolaihu/Downloads/test/feature'
    cFe = os.path.join('/home/luolaihu/Downloads/test', 'efeature')
    featureE(baseFe, cFe)

def testFe():
    CFEATUREMAP = dict()
    LFEATUREMAP = dict()
    DFEATUREMAP = dict()
    fileName = '/home/luolaihu/Downloads/test/feature'
    with open(fileName, 'r') as f:
        for line in f:
            items = line.strip().split('\001')
            if len(items) == 3:
                CFEATUREMAP[items[0]] = items[2]
                LFEATUREMAP[items[0]] = items[1].split(',')[0]
                DFEATUREMAP[items[0]] = items[1].split(',')[1]
            else:
                print('format error : %' % line)
    print("load feature finish CFEATUREMAP : %d " % len(CFEATUREMAP))
    print("load feature finish LFEATUREMAP : %d " % len(LFEATUREMAP))

    featureDict = dict()
    with open('/home/luolaihu/Downloads/train/featuredict' , 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                featureDict = json.loads(line)

    from preprocess import datatransform as df
    baseDir = '/home/luolaihu/Downloads/season_1/test_set_1'
    df.mapInit(baseDir)

    predictTime = set()
    with open('/home/luolaihu/Downloads/season_1/test_set_1/read_me_1.txt') as tt:
        for line in tt:
            line = line.strip()
            if line.startswith('2'):
                predictTime.add(line)

    keyList = list()
    for i in range(1,67):
        for key in predictTime:
            keyList.append(str(i) + '#' + key)

    outPath = os.path.join('/home/luolaihu/Downloads/test', 'tfeature')
    if os.path.exists(outPath):
        os.remove(outPath)

    labelPath = os.path.join('/home/luolaihu/Downloads/test', 'labelMap')
    if os.path.exists(labelPath):
        os.remove(labelPath)

    with open(outPath, 'a') as out:
        for key in sorted(keyList):
            feStr = ''
            labelStr = ''
            destStr = ''
            label = ''
            if key in LFEATUREMAP:
                label = LFEATUREMAP[key]
            else:
                feStr = df.getFeature(key, featureDict) + '\t'

            for i in range(0, COMBINEFE):
                items = key.split('-')
                if len(items) == 4:
                    minute = int(items[3])
                    day = items[2]
                    month = items[1]
                    year = items[0].split('#')[1]
                    clid = items[0].split('#')[0]
                    minute = minute - i
                    dateStr = year + '-' + month + '-' + day
                    sliceTime = datetime.datetime.strptime(dateStr, '%Y-%m-%d')
                    deltaMinute = datetime.timedelta(minutes=i * -10)
                    if minute <= 0:
                        minute = minute + 144
                        dateStr = datetime.datetime.strftime(sliceTime + deltaMinute, '%Y-%m-%d')
                    ikey = clid + '#' + dateStr + '-' + str(minute)
                    if i == 4:
                        print('ikey %s : key %s' % (ikey, key))
                    if ikey in CFEATUREMAP:
                        fe = CFEATUREMAP[ikey]
                        fitems = fe.split('\t')
                        for fitem in fitems:
                            kvs = fitem.split(':')
                            fid = int(kvs[0])
                            fid = fid + i * 185
                            nkvs = str(fid) + ':' + kvs[1]
                            feStr = feStr + nkvs + '\t'
                    # print('key %s : lastStr %s' % (ikey, feStr))
                    if ikey in LFEATUREMAP:
                        k = COMBINEFE * 185 + i
                        fe = LFEATUREMAP[ikey]
                        if i == 0 :
                            fe = '0'
                        labelStr = labelStr + str(k) + ':' + fe + '\t'

                    if ikey in DFEATUREMAP:
                        k = COMBINEFE * 186 + i
                        fe = DFEATUREMAP[ikey]
                        if i == 0:
                            fe = '0'
                        destStr = destStr + str(k) + ':' + fe + '\t'
            out.write(label + '\t' + feStr.strip() + '\t' + labelStr.strip() + '\t' + destStr.strip() + '\r\n')
            with open(labelPath, 'a') as labelOut:
                labelOut.write(key + '#' + label + '\r\n')

if __name__ == '__main__':
    trainFe()
    evalFe()
    testFe()