# data=[116.0, 93.0, 67.0, 57.0, 63.0, 59.0, 50.0, 42.0, 33.0, 31.0, 33.0, 39.0, 37.0, 38.0, 36.0, 34.0, 37.0, 35.0, 35.0, 33.0, 34.0, 28.0, 26.0, 30.0, 33.0, 35.0, 33.0, 32.0, 29.0, 48.62, 25.0, 25.0, 24.0, 27.0, 26.0, 20.0, 23.0, 27.0, 30.0, 24.0, 15.0, 11.0, 13.0, 7.0, 16.0, 20.0, 28.0, 32.0, 25.0, 28.0, 29.0, 33.0, 39.0, 32.0, 27.0, 32.0, 26.0, 20.0, 20.0, 18.0, 18.0, 15.0, 9.0, 10.0, 12.0, 10.0, 14.0, 8.0, 6.0, 7.0, 10.0, 8.0, 10.0, 11.0, 9.0, 15.0, 10.0, 9.0, 10.0, 7.0, 7.0, 9.0, 9.0, 9.0, 6.0, 4.0, 7.0, 5.0, 2.0, 48.62, 5.0, 4.0, 7.0, 7.0, 13.0, 15.0, 19.0, 48.62, 21.0, 23.0, 28.0, 30.0, 36.0, 42.0, 43.0, 51.0, 51.0, 54.0, 66.0, 84.0, 90.0, 84.0, 73.0, 78.0, 74.0, 68.0, 74.0, 78.0, 77.0, 81.0, 86.0, 86.0, 59.0, 53.0, 50.0, 52.0, 55.0, 66.0]
#
# print(len(data))
#
# import matplotlib.pyplot as plt
# plt.figure()
# observed= [116.0, 93.0, 67.0, 57.0, 63.0, 59.0, 50.0, 42.0, 33.0, 31.0, 33.0, 39.0, 37.0, 38.0, 36.0, 34.0, 37.0, 35.0, 35.0, 33.0,
#            34.0, 28.0, 26.0, 30.0, 33.0, 35.0, 33.0, 32.0, 29.0, 48.62, 25.0, 25.0, 24.0, 27.0, 26.0, 20.0, 23.0, 27.0, 30.0, 24.0,
#            22.0,22,21,23,25,28,35,29]
#
# observed1=[7.0, 9.0, 9.0, 9.0, 6.0, 4.0, 7.0, 5.0, 2.0, 48.62, 5.0, 4.0, 7.0, 7.0, 13.0, 15.0, 19.0, 48.62, 21.0, 23.0, 28.0, 30.0,
#            36.0, 42.0, 43.0, 51.0, 51.0, 54.0, 66.0, 84.0, 90.0, 84.0, 73.0, 78.0, 74.0, 68.0, 74.0, 78.0, 77.0, 81.0,
#            80,83,87,88,79,72,72,71]
#
#
# pre= [117, 115, 70,78,68,49,29,38,42,55,41,42,54,42,29,40,40,39,59,40,
#      39,51,59,59,69,60,40,50,51,78,61,70,62,71,40,45,49,39,40,39,
#      50,60,62,67,68,73,78,89 ]
#
# pre1=[8, 9, 8, 6, 8 ,14, 11, 15, 11,60,41,32,24,31,45,40,60,79,59,40,
#       39,59,49,89,59,65,80,65,71,71,81,90,100,103,112,111,132,115,119,133,
#       104,113,117,119,120,123,121,120]
#
# predicted= [117, 92, 60, 52, 60, 50, 46, 32,28,27,37,36,27,26,38,49,58,48,46,49,
#             36,34,34,35,46,78,68,60,50,59,50,41,29,31,40,30,32,43,54,47,
#             50,54,59,65,70,85,87,76]
#
# predicted1=[7, 8.8, 9 , 9, 6,  4, 6, 8, 8, 49, 7, 9, 11, 10, 16, 17,22,70, 26,28,
#             38,44,49,50,56,57,61,69,56,64,64,71,71,64,57,41,51,60,21,32,
#             30,37,40,45,35,34,30,38]
#
#
#
#
# plt.subplot(2, 1, 1)
# # plt.scatter(range(40),predicted1)
# plt.plot(predicted,color='black',marker='.',linewidth=1.5,label='FDN-Learning')
# plt.plot(pre,color='b',marker='*',linewidth=1,label='CNN-AEPP')
# plt.plot(observed,color='r',marker='o',linewidth=1,label='Observed value')
# font2 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 13,
# }
# plt.ylabel('PM2.5(ug/m3)',font2)
# # plt.xlabel('Time(h)',font2)
# plt.title('Predicted and observed values for PM2.5',font2)
# font1 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 13,
# }
# plt.legend(loc='upper right',prop=font1)
# plt.grid()
#
#
# plt.subplot(2, 1, 2)
# plt.plot(predicted1,color='black',marker='.',linewidth=1.5)
# plt.plot(pre1,color='b',marker='*',linewidth=1)
# plt.plot(observed1,color='r',marker='o',linewidth=1)
# font2 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 13,
# }
# plt.ylabel('PM2.5(ug/m3)',font2)
# plt.xlabel('Time(h)',font2)
# # plt.title('Predicted and observed values for PM2.5',font2)
#
# font1 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 13,
# }
# # plt.legend(loc='upper center',prop=font1)
# plt.grid()
# plt.show()

import matplotlib.pyplot as plt
plt.figure()

observed=[95.0, 97.0, 100.0, 105.0, 101.0, 90.0, 77.0, 68.0, 68.0, 68.0, 64.0, 61.0, 56.0, 51.0, 45.0, 41.0, 39.0, 38.0, 39.0, 38.0, 39.0, 39.0, 37.0, 35.0]

RNN=[47.846992, 43.335255, 39.193424, 35.616734, 32.738548, 30.5776, 29.037388, 27.980738, 77.341637, 78.873688, 76.297539, 76.187828, 74.385529, 72.380829, 70.444748, 68.461311, 66.378334, 64.181648, 61.864708, 59.438229, 56.919605, 54.327572, 51.682873, 49.012512]


GRU=[21.320042, 21.201075, 21.125933, 21.081367, 21.053959, 21.035957, 21.023397, 21.014118, 74.764099, 77.158424, 79.018387, 80.410645, 81.639763, 82.627235, 83.398453, 84.003876, 84.438347, 84.702507, 84.83419, 84.887848, 84.908699, 84.926582, 84.959106, 85.015785]


LSTM=[30.859083, 30.743338, 30.661716, 30.607565, 30.575344, 30.559319, 30.555122, 30.559612, 35.937115, 73.096985, 78.311523, 77.453018, 75.489021, 74.405762, 73.962982, 73.748894, 73.556679, 73.295815, 72.938255, 72.483475, 71.939056, 71.313141, 70.612389, 69.841766]

EDSModel=[56.526756, 56.541229, 55.650101, 54.526283, 53.884308, 54.231583, 55.819458, 58.694145, 59.742226, 56.503769, 53.73455, 53.49712, 53.81255, 55.476543, 58.171078, 61.597454, 65.615257, 69.850922, 73.687149, 76.379929, 77.266808, 76.08683, 73.192886, 69.527588]


#plt.plot(BP,'m-',linewidth=1,label='BP')
plt.plot(observed[0:],color='b',linewidth=1.2,marker='.',label='Observed value')
plt.plot(RNN[0:],color='black',linewidth=1.2,marker='*',label='RNN-Encoder-Decoder')
#plt.plot(MLR,color='y',linewidth=1,label='MLR')
#plt.plot(RNN,color='blue',linewidth=1,label='RNN')
plt.plot(GRU[0:],color='orange',marker='*',linewidth=1.2,label='GRU-Encoder-Decoder')
#
plt.plot(LSTM[0:],color='y',linewidth=1.2,marker='*',label='LSTM-Encoder-Decoder')
#plt.plot(CNN_ConvLSTM,color='navy',linewidth=1,label='CNN+ConvLSTM')
#plt.plot(ResNet_LSTM,color='lightcoral',linewidth=1,label='ResNet+LSTM')
EDSModel.reverse()
plt.plot(EDSModel[0:24],color='red',linewidth=1.2,marker='*',label='EDSModel')
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 13,
}
plt.ylabel('PM2.5(ug/m3)',font2)
plt.xlabel('Prediction time',font2)
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 13,
}
plt.legend(loc='upper left',prop=font1)
plt.grid()
plt.show()