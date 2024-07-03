import streamlit as st
import pandas as pd
import numpy as np
import pickle

model=pickle.load(open("RandomForest.pkl","rb"))
st.title("Crop Prediction App")
col1,col2,col3=st.columns(3)
with col1:
    N=st.number_input("Enter Nitrogen")
with col2:
    P=st.number_input("Enter phosphorus")
with col3:
    K=st.number_input("Enter potassium")
col4,col5,col6,col7=st.columns(4)
with col4:
    temperature=st.number_input("Enter Temperature")
with col5:
    humidity=st.number_input("humidity")
with col6:
    ph=st.number_input("ph")
with col7:
    rainfall=st.number_input("Enter rainfall")
input_features=pd.DataFrame({
    'N':[N],
    'P':[P],
    'K':[K],
    'temperature':[temperature],
    'humidity':[humidity],
    'ph':[ph],
    'rainfall':[rainfall]
})
if st.button("predict crop"):
    result=model.predict(np.array(input_features))
    st.text(result[0])
