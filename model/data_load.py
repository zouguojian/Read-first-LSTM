import pandas as pd
import csv
a='/Users/guojianzou/Documents/20_city_data/shanghai/2014.csv'
b='/Users/guojianzou/Documents/20_city_data/shanghai/train.csv'
c='/Users/guojianzou/Documents/20_city_data/shanghai/test.csv'

D='/Users/guojianzou/Documents/20_city_data/2019/苏州.csv'

columns = ['time', 'AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h', 'NO2', 'NO2_24h',
               'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h']
def writerData(address1):
    fil=open(address1,'w',newline='')
    write=csv.writer(fil)
    # file=pd.read_csv(address,'r').values
    write.writerow(columns)
    for i in range(4,8):
        file=pd.read_csv('/Users/guojianzou/Documents/20_city_data/shanghai/201'+str(i)+'.csv','r').values
        for line in file:
            row=[]
            for char in line[0].split(','):
                if len(row)>=1:row.append(float(char))
                else:row.append(char)
            print(row)
            write.writerow(row)
    fil.close()
# writerData(b)
file=pd.read_csv(D,usecols=['time','AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h', 'NO2', 'NO2_24h',
               'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h'])
file.fillna(method='ffill',inplace=True)

fil = open(D, 'w', newline='')
write = csv.writer(fil)
write.writerow(columns)
write.writerows(file.values[2156:])
fil.close()

print(file.values[2156])