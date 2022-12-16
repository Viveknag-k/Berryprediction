import streamlit as st
import pandas as pd
import numpy as np

#import joblib
#model = joblib.load('linear_regrs.joblib')

import pickle
model=pickle.load(open("lr.pkl","rb"))
st.header(" Blueberry yield Prediction")
st.write("Input Parameters")
def ip_features():
    #[37.5,42.5,0.75,0.50]
    #[0.75,0.50,0.25,1]
    clonesize=st.number_input("clonesize")
    honeybee=st.number_input("honeybee")
    bumbles=st.number_input("bumbles")
    andrena=st.number_input("andrena")
    osmia=st.number_input("osmia")
    data = {'clonesize':clonesize,
            'honeybee':honeybee,
            'bumbles':bumbles,
            'andrena':andrena,
            'osmia':osmia}
    features= pd.DataFrame(data,index=[0])
    return features
df = ip_features()
st.write(df)

DF = pd.read_csv("C:\\Users\\Vivek Nag Kanuri\\Downloads\\WildBlueberryPollinationSimulationData.csv")
x = DF[['clonesize','honeybee','bumbles','andrena','osmia']]
#x=DF(features)
y= DF['yield']

model.fit(x,y)

pred = model.predict(df)
st.write('yield',pred)


