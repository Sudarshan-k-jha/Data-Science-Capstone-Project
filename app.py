from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
model = pickle.load(open('dtr.pkl', 'rb'))
    Fuel_Type_Diesel=1
    if request.method == 'POST':
        Year = int(request.form['year'])
        selling_price=float(request.form['selling_price'])
        km_driven=float(request.form['km_driven'])
        owner_First Owner=request.form['owner_First Owner']
        if(owner_First Owner=='First owner'):
            owner_First Owner=0
            owner_Second Owner=2
            owner_Third Owner=3
            owner_Fourth & Above Owner=1
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
        prediction=model.predict([[selling_price,km_driven2,owner_First Owner,owner_Second Owner,owner_Third Owner,owner_Fourth & Above Owner,owner_Test Drive Cars,Year,fuel_Diesel,fuel_Petrol,fuel_CNG,fuel_LPG,fuel_Electric,seller_type_Individual,seller_type_Dealer,seller_type_Trustmark Dealer,transmission_Manual,transmittion_Automatic]])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index.html',prediction_text="You Can Sell The Car at {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
