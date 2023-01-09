import streamlit as st
import pickle
import numpy as np
from sklearn import *

# import the model
df = pickle.load(open('df.pkl', 'rb'))
df1 = pickle.load(open('df1.pkl','rb'))
dtr = pickle.load(open('dtr.pkl','rb'))

st.title("Car Predictor")

name = st.selectbox('name',df1['name'].unique())
year = st.selectbox('year',df1['year'])
selling_price=float(request.form['selling_price'])
km_driven=float(request.form['km_driven'])
owner_First Owner=request.form['owner_First Owner']
if(owner_First Owner=='First owner'):
        owner_First owner=0
        owner_Second owner=2
        owner_Third owner=3
        owner_Fourth & Above owner=1
         else:
            owner_Test Drive Car=4
        fuel_Petrol=request.form['fuel_Petrol']
        if(fuel_Petrol=='Petrol'):
                fuel_Petrol=4
                fuel_Diesel=1
                fuel_CNG=0
                fuel_LPG=3      
        else:
            fuel_Electric=2
        Year=2023-year
        seller_type_Individual=request.form['seller_type_Individual']
        if(seller_type_Individual=='Individual'):
            seller_type_Individual=1
            seller_type_Dealer=0
        else:
            seller_type_Trustmark Dealer=2	
        transmission_Manual=request.form['transmission_Mannual']
        if(transmission_Manual=='Manual'):
            transmission_Manual=1
        else:
            transmission_Automatic=0
        prediction=model.predict([[selling_price,km_driven,owner_First Owner,owner_Second Owner,owner_Third Owner,owner_Fourth & Above Owner,owner_Test Drive Cars,Year,fuel_Diesel,fuel_Petrol,fuel_CNG,fuel_LPG,fuel_Electric,seller_type_Individual,seller_type_Dealer,seller_type_Trustmark Dealer,transmission_Manual,transmittion_Automatic]])
        output=round(prediction[0],2)
