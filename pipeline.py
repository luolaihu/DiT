#!/usr/bin/python3
# -*- coding:utf8 -*-
# @author luolaihu

from preprocess import datatransform
from preprocess import featurecombine
from trainmodel import regression
from predict import predict

def main():
    datatransform.run()
    featurecombine.run()
    regression.train()
    # predict.predict()

if __name__ == '__main__':
    main()