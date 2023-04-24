import pandas as pd
import requests

url='https://www.twse.com.tw/exchangeReport/BWIBBU_d?response=csv&date=20211210&selectType=ALL'
data=requests.get(url)

#print(data.text.replace("\"",""))
#證券代號,證券名稱,殖利率(%),股利年度,本益比,股價淨值比,財報年/季,

data_list=[]

for row in data.text.replace("\"","").split("\n"):
    rows=row.split(",")
    if len(rows)==8 and rows[0]!='證券代號':
        #print(row)
        d={
            '證券代號': rows[0],
            '證券名稱': rows[1],
            '殖利率':None if rows[2] == '-' else float(rows[2]),
            '股利年度': rows[3],
            '本益比': None if rows[4] == '-' else float(rows[4]),
            '股價淨值比': None if rows[5] == '-' else float(rows[5]),
            '財報年/季': rows[6]

        }
        data_list.append(d)

#print(data_list)
df=pd.DataFrame(data_list)
# print(df.head())
i=df.get("殖利率") > 10
j=df.get("本益比") < 10
k=df.get("股價淨值比").between(0,1)
print(df[i & j & k])
