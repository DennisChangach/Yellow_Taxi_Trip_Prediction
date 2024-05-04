import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler


#Function to get input info
def trip_duration_form():
    with st.form(key='duration',clear_on_submit=False):
        st.subheader("Predict Trip Duration 🕒")
        tpep_pickup_datetime = st.text_input("pickup_datetime",value="2017-07-27 18:31:15")
        DOLocationID = st.number_input("DOLocationID",value=246)
        RatecodeID= st.selectbox("RatecodeID",[1,2,3,4,5,6],index=0)
        payment_type = st.selectbox("payment_type",[0,1,2,3,4,5,6],index=2)
        trip_distance = st.number_input("trip_distance",value=2.25)
        fare_amount = st.number_input("fare_amount",value = 13.0)
        extra = st.number_input("extra",value=1.0)
        if st.form_submit_button("Predict 🔮"):
            #columns
            columns = ['tpep_pickup_datetime','trip_distance', 'RatecodeID', 'DOLocationID', 'payment_type',
       'fare_amount', 'extra']
            values = [tpep_pickup_datetime,trip_distance, RatecodeID, DOLocationID, payment_type,
       fare_amount, extra]
            with st.spinner("Predicting..."):
                features_trip_duration(columns,values)

#Function to get input info
def trip_fare_form():
    with st.form(key='fare_amount',clear_on_submit=False):
        st.subheader("Predict Taxi Fare 💵")
        VendorID= st.selectbox("VendorID",[1,2],index=1)
        tpep_pickup_datetime = st.text_input("Pickup_datetime",value="2017-07-27 18:31:15")
        tpep_dropoff_datetime = st.text_input("Dropoff_datetime",value="2017-07-27 18:48:44")
        DOLocationID = st.number_input("DOLocationID",value=246)
        trip_distance = st.number_input("trip_distance",value=2.25)
        RatecodeID= st.selectbox("RatecodeID",[1,2,3,4,5,6],index=0)
        payment_type = st.selectbox("payment_type",[0,1,2,3,4,5,6],index=1)
        extra = st.number_input("extra",value=1.0)
        mta_tax = st.number_input("mta_tax",value=0.5)
        tip_amount = st.number_input("tip_amount",value=0.0)
        tolls_amount = st.number_input("tolls_amount",value=0.0)
        improvement_surcharge = st.number_input("improvement_surcharge",value=0.3)

        if st.form_submit_button("Predict 🔮"):
            columns = ['VendorID', 'tpep_pickup_datetime',
       'tpep_dropoff_datetime',  'DOLocationID','trip_distance',
       'RatecodeID','payment_type', 'extra', 'mta_tax', 'tip_amount',
       'tolls_amount', 'improvement_surcharge']
            values = [VendorID, tpep_pickup_datetime,
       tpep_dropoff_datetime,  DOLocationID,trip_distance,
       RatecodeID,payment_type, extra, mta_tax, tip_amount,
       tolls_amount, improvement_surcharge]
             #Additioal charges
            added_charges = extra+mta_tax+tip_amount+tolls_amount+improvement_surcharge
            with st.spinner("Predicting..."):
                features_trip_fare(columns,values,added_charges)
            

#Function to feature engineer: trip duration features
def features_trip_duration(columns,values):
    # Create a dictionary with column names as keys and lists as values
    series = pd.Series(values,index=columns)
    # Create the DataFrame
    df = pd.DataFrame([series])
    #convert to datetime
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    #Get new features
    df['pickup_day_no']=df['tpep_pickup_datetime'].dt.weekday
    df['pickup_hour']=df['tpep_pickup_datetime'].dt.hour

    #Drop pickup time
    df.drop(['tpep_pickup_datetime'],axis=1,inplace=True)
  
    #st.dataframe(df)
    
    #scaling features
    df2 = feature_scaler_duration(df)
    #st.dataframe(df2)

    #making trip duration prediction
    predict_trip_duration(df2)

#Function for scaling
def feature_scaler_duration(df):
    #Adding missing features
    all_columns = set(['VendorID', 'passenger_count', 'trip_distance', 'RatecodeID',
       'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra',
       'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
       'total_amount', 'pickup_day_no', 'pickup_hour', 'pickup_month',
       'store_and_fwd_flag_Y'])
    
    columns = set(df.columns)

    new_cols = list(all_columns.difference(columns))

    #st.write(new_cols)
    for col in new_cols:
        df[col] = 0
    
    #reararange
    df = df[['VendorID', 'passenger_count', 'trip_distance', 'RatecodeID',
       'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra',
       'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
       'total_amount', 'pickup_day_no', 'pickup_hour', 'pickup_month',
       'store_and_fwd_flag_Y']]
    #st.dataframe(df)
    
    # Load the saved scaler from the pickle file
    with open("..\Models\duration_scaler.pickle", "rb") as f:
        scaler = pickle.load(f)
    df2 = pd.DataFrame(scaler.transform(df),columns=df.columns)

    #After scaling--select the features meeded
    df2 = df2[['trip_distance', 'RatecodeID', 'DOLocationID', 'payment_type',
       'fare_amount', 'extra', 'pickup_day_no', 'pickup_hour']]

    return df2


#Function to predict trip duration
def predict_trip_duration(df):
    # Path to your pickled model file
    model_path = "..\Models\predict_trip_duration.pickle"

    # Load the pickled model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict(df)
    st.write(f"The predicted trip duration is: {round(prediction[0],2)} minutes")

#Function to feature engineer: trip duration features
def features_trip_fare(columns,values,added_charges):
    # Create a dictionary with column names as keys and lists as values
    series = pd.Series(values,index=columns)
    # Create the DataFrame
    df = pd.DataFrame([series])
    #st.dataframe(df)
    #convert to datetime
    #Converting to datetime
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    #Get new features
    #calculating trip duration(in minutes) using pickup & dropoff times
    df['trip_duration'] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    df['pickup_day_no']=df['tpep_pickup_datetime'].dt.weekday
    df['dropoff_hour']=df['tpep_dropoff_datetime'].dt.hour
    df['pickup_month']=df['tpep_pickup_datetime'].dt.month
    df['dropoff_month']=df['tpep_dropoff_datetime'].dt.month

    #Drop the pickup time stamp
    df.drop(['tpep_pickup_datetime','tpep_dropoff_datetime'],axis=1, inplace=True)
    
    #rearrange the columns to match
    df = df[['VendorID', 'trip_distance', 'RatecodeID', 'DOLocationID',
       'payment_type', 'trip_duration', 'pickup_day_no', 'dropoff_hour',
       'pickup_month', 'dropoff_month']]
    #st.dataframe(df)

    df2 = feature_scaler_amount(df)
    #making trip duration prediction
    predict_taxi_fare(df2,added_charges)


#Function for scaling
def feature_scaler_amount(df):
    #Adding missing features
    all_columns = set(['VendorID', 'passenger_count', 'trip_distance', 'RatecodeID',
       'PULocationID', 'DOLocationID', 'payment_type', 'trip_duration',
       'pickup_day_no', 'dropoff_day_no', 'pickup_hour', 'dropoff_hour',
       'pickup_month', 'dropoff_month', 'pickup_year', 'dropoff_year',
       'store_and_fwd_flag_Y'])
    
    columns = set(df.columns)

    new_cols = list(all_columns.difference(columns))

    #st.write(new_cols)
    for col in new_cols:
        df[col] = 0
    
    #reararange
    df = df[['VendorID', 'passenger_count', 'trip_distance', 'RatecodeID',
       'PULocationID', 'DOLocationID', 'payment_type', 'trip_duration',
       'pickup_day_no', 'dropoff_day_no', 'pickup_hour', 'dropoff_hour',
       'pickup_month', 'dropoff_month', 'pickup_year', 'dropoff_year',
       'store_and_fwd_flag_Y']]
    #st.dataframe(df)
    
    # Load the saved scaler from the pickle file
    with open("..\Models\\fare_scaler.pickle", "rb") as f:
        scaler = pickle.load(f)
    df2 = pd.DataFrame(scaler.transform(df),columns=df.columns)

    #After scaling--select the features meeded
    df2 = df2[['VendorID', 'trip_distance', 'RatecodeID', 'DOLocationID',
       'payment_type', 'trip_duration', 'pickup_day_no', 'dropoff_hour',
       'pickup_month', 'dropoff_month']]

    return df2

#Function to predict trip duration
def predict_taxi_fare(df,added_charges):
    # Path to your pickled model file
    model_path = "..\Models\predict_taxi_fare.pickle"

    # Load the pickled model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict(df)
    st.write(f"The predicted fare amount is: {prediction[0]} thus the total amount charged is {round(prediction[0]+added_charges,2)}")


