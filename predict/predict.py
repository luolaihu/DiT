#!/usr/bin/python3
# -*- coding:utf8 -*-
# @author luolaihu

import xgboost as xgb
import os

def predict():
    modelPath = '/home/luolaihu/Downloads/model/0001.model'
    bst = xgb.Booster({'nthread': 4}, model_file=modelPath)
    dtest = xgb.DMatrix('/home/luolaihu/Downloads/test/tfeature')
    indexMap = dict()
    i = 0
    with open('/home/luolaihu/Downloads/test/labelMap', 'r') as f:
        for line in f:
            indexMap[i] = line.strip()
            i = i + 1
    preds = bst.predict(dtest, ntree_limit = 15)

    resultPath = '/home/luolaihu/Downloads/test/result.csv'
    if os.path.exists(resultPath):
        os.remove(resultPath)
    with open(resultPath, 'a') as out:
        for i in range(len(preds)):
            out.write('%s%s' % (indexMap[i].replace('#', ','), preds[i]) + '\r\n')

if __name__ == '__main__':
    predict()