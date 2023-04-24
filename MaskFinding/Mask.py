import json
from math import radians, cos, sin, asin, sqrt

import pandas as pd
import requests


#大園距離公式
def distance(lon1, lat1, lon2, lat2):
    # 轉換弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # 距離公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半徑，單位公里
    return c * r * 1000

url = 'https://raw.githubusercontent.com/kiang/pharmacies/master/json/points.json'
jsonObj=json.loads(requests.get(url).text)
# print(jsonObj)
data_list=[]
lat,lng=25.035922, 121.533653
for data in jsonObj['features']:
    d={
        'name':data['properties']['name'],
        'mask_adult': data['properties']['mask_adult'],
        'mask_child': data['properties']['mask_child'],
        'lng': data['geometry']['coordinates'][0],
        'lat': data['geometry']['coordinates'][1],
        'm': distance(lng, lat, data['geometry']['coordinates'][0], data['geometry']['coordinates'][1]),
        'address': data['properties']['address']
    }
    data_list.append(d)


# print(data_list)

df=pd.DataFrame(data_list)
i=df.get('mask_adult') > 300
j=df.get('mask_child') > 300
k=df.get('m') < 500
df=df[i & j & k]
print(df[['name','m','mask_adult','mask_child','address']])