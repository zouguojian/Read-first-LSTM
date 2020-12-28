# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:50:59 2017

@author: Administrator
"""
import pandas as pd
import numpy as np
def Dataset():
    cols = ['AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h', 'NO2', 'NO2_24h',
               'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h']
    test = pd.read_csv('/Users/guojianzou/Documents/20_city_data/shanghai/2018.csv',usecols=cols)
    test.fillna(method='ffill',inplace=True)
    return test.values

#获取测试集的数据以及数据集大小
data=Dataset()
length=data.shape[0]

#每次测试窗口移动的步长
step=24

def train_data(batch_size,time_size,prediction_size):
    SH_train_low=0
    SH_train_hight = SH_train_low + batch_size * time_size
    while(SH_train_hight+prediction_size<length):
        label=list()
        for line in range(batch_size):
            for time in range(prediction_size):
                    label.append(data[SH_train_low + time + time_size * (line + 1), 1])
        yield np.reshape(data[SH_train_low:SH_train_hight,:],[batch_size,time_size,-1]),np.reshape(label,[batch_size,prediction_size])
        SH_train_low = SH_train_low+step
        SH_train_hight = SH_train_low + batch_size * time_size