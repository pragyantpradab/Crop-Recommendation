# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 00:32:10 2023

@author: Gokul
"""
import pandas as pd
from geopy.distance import great_circle
df_final=pd.read_csv(r'E:\781 project datasets\Dataset for crop recommendation.csv')
import pickle
with open("E:\Downloads_2\knn_model.pkl", 'rb') as model_file:
    knn = pickle.load(model_file)
from geopy.distance import great_circle
def getpredictors(latitude,longitude):
    Lat=latitude
    Long=longitude
    latlong_user=(Lat,Long)
    dist={}
    for i in range(len(df_final)):
      D=df_final.iloc[i].to_dict()
      L=list(D.values())[9:11]
      place_coord=(L[0],L[1])
      dist[D['Dist Name']]=great_circle(latlong_user, place_coord).km
    dist={k: v for k, v in sorted(dist.items(), key=lambda item: item[1])}
    Near3=list(dist.keys())[0:3]
    dist3=list(dist.values())[0:3]
    df_1nearest = df_final[df_final['Dist Name'] == Near3[0]]
    df_2nearest= df_final[df_final['Dist Name'] == Near3[1]]
    df_3nearest= df_final[df_final['Dist Name'] == Near3[2]]
    df_nearest = pd.concat([df_1nearest, df_2nearest, df_3nearest], ignore_index=True)
    X_feed=pd.DataFrame(columns=['Annual Rainfall(mm)','NITROGEN SHARE IN NPK (Percent)','PHOSPHATE SHARE IN NPK (Percent)','POTASH SHARE IN NPK (Percent)','Avg Max Temp','Avg Min Temp',"Soil_INCEPTISOLS ","Soil_ORTHIDS ","Soil_PSAMMENTS ","Soil_SANDY ALFISOL ","Soil_USTALF/USTOLLS ","Soil_VERTIC SOILS "])
    X_feed['Soil_type']=[list(df_nearest['Soil_type'])[0],]
    X_feed.iloc[0] = X_feed.iloc[0].fillna(0)
    soiltype=list(X_feed.iloc[0].to_dict().values())[12]
    soil="Soil_"+soiltype
    X_feed[soil]=1
    def avg(L):
      return sum(L)/len(L)
    def wtavg(a,b,c):
      return ((a/dist3[0])+(b/dist3[1])+(c/dist3[2]))/((1/dist3[0])+(1/dist3[1])+(1/dist3[2]))
    avg_1_rainfall=avg(list(df_1nearest['Annual Rainfall(mm)']))
    avg_2_rainfall=avg(list(df_2nearest['Annual Rainfall(mm)']))
    avg_3_rainfall=avg(list(df_3nearest['Annual Rainfall(mm)']))
    avg_1_N=avg(list(df_1nearest['NITROGEN SHARE IN NPK (Percent)']))
    avg_2_N=avg(list(df_2nearest['NITROGEN SHARE IN NPK (Percent)']))
    avg_3_N=avg(list(df_3nearest['NITROGEN SHARE IN NPK (Percent)']))
    avg_1_P=avg(list(df_1nearest['PHOSPHATE SHARE IN NPK (Percent)']))
    avg_2_P=avg(list(df_2nearest['PHOSPHATE SHARE IN NPK (Percent)']))
    avg_3_P=avg(list(df_3nearest['PHOSPHATE SHARE IN NPK (Percent)']))
    avg_1_K=avg(list(df_1nearest['POTASH SHARE IN NPK (Percent)']))
    avg_2_K=avg(list(df_2nearest['POTASH SHARE IN NPK (Percent)']))
    avg_3_K=avg(list(df_3nearest['POTASH SHARE IN NPK (Percent)']))
    avg_1_Tmax=avg(list(df_1nearest['Avg Max Temp']))
    avg_2_Tmax=avg(list(df_2nearest['Avg Max Temp']))
    avg_3_Tmax=avg(list(df_3nearest['Avg Max Temp']))
    avg_1_Tmin=avg(list(df_1nearest['Avg Min Temp']))
    avg_2_Tmin=avg(list(df_2nearest['Avg Min Temp']))
    avg_3_Tmin=avg(list(df_3nearest['Avg Min Temp']))
    X_feed['Annual Rainfall(mm)']=[wtavg(avg_1_rainfall,avg_2_rainfall,avg_3_rainfall),]
    X_feed['NITROGEN SHARE IN NPK (Percent)']=[wtavg(avg_1_N,avg_2_N,avg_3_N),]
    X_feed['PHOSPHATE SHARE IN NPK (Percent)']=[wtavg(avg_1_P,avg_2_P,avg_3_P),]
    X_feed['POTASH SHARE IN NPK (Percent)']=[wtavg(avg_1_K,avg_2_K,avg_3_K),]
    X_feed['Avg Max Temp']=[wtavg(avg_1_Tmax,avg_2_Tmax,avg_3_Tmax),]
    X_feed['Avg Min Temp']=[wtavg(avg_1_Tmin,avg_2_Tmin,avg_3_Tmin),]
    X_feed.drop(columns=["Soil_type"],inplace=True)
    pred=list(knn.predict(X_feed))[0]
    return pred