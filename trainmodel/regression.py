#!/usr/bin/python3
# -*- coding:utf8 -*-

import os

import xgboost as xgb


def train():
    modelPath = '/home/luolaihu/Downloads/model/0001.model'
    dtrain = xgb.DMatrix('/home/luolaihu/Downloads/train/cfeature')
    deval = xgb.DMatrix('/home/luolaihu/Downloads/test/efeature')
    param = {'max_depth':4, 'eta':0.2, 'silent':1, 'objective':'reg:linear',
             'min_child_weight' : 1, 'gamma': 0.1, 'nthread': 4,
             'subsample':0.7,'lambda':1}
    # 'booster':'gblinear', 'lambda':1, 'alpha':1
    # , 'colsample_bytree':0.7
    # watchlist = [(deval, 'eval'), (dtrain, 'train')]
    watchlist = [(deval, 'eval')]
    num_round = 100
    bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=10)
    print('Train Finish!')
    if os.path.exists(modelPath):
        os.remove(modelPath)
    bst.save_model(modelPath)
    # dump model
    dumpPath = '/home/luolaihu/Downloads/model/dump.raw.txt'
    if os.path.exists(dumpPath):
        os.remove(dumpPath)
    bst.dump_model(dumpPath)

    print('best round: %d' % bst.best_ntree_limit)
    best_round = bst.best_ntree_limit

    bst = xgb.Booster({'nthread': 4}, model_file=modelPath)
    deval = xgb.DMatrix('/home/luolaihu/Downloads/test/efeature')
    predEvals = bst.predict(deval, ntree_limit = best_round)
    labels = deval.get_label()
    for i in range(1, len(predEvals)):
        print('%s:%s' % (labels[i], predEvals[i]))
    print('MAPE : %f' % float(sum(abs(predEvals[i] - labels[i])/labels[i] for i in range(1,len(predEvals)) if labels[i] > 0.0)/len(labels > 0.0)))
    # xgb.plot_importance(bst)

    print('Get Submit!')
    bst = xgb.Booster({'nthread': 4}, model_file=modelPath)
    dtest = xgb.DMatrix('/home/luolaihu/Downloads/test/tfeature')
    indexMap = dict()
    i = 0
    with open('/home/luolaihu/Downloads/test/labelMap', 'r') as f:
        for line in f:
            indexMap[i] = line.strip()
            i = i + 1
    preds = bst.predict(dtest, ntree_limit = best_round)

    resultPath = '/home/luolaihu/Downloads/test/result.csv'
    if os.path.exists(resultPath):
        os.remove(resultPath)
    with open(resultPath, 'a') as out:
        for i in range(len(preds)):
            out.write('%s%s' % (indexMap[i].replace('#', ','), preds[i]) + '\r\n')
            # print('%s%s' % (indexMap[i].replace('#', ','), preds[i]))

if __name__ == '__main__':
    train()