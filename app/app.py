import streamlit as st
import numpy as np
import pandas as pd

from utils import trip_duration_form,trip_fare_form


def main():
    st.set_page_config(page_title = "Caantin AI",page_icon = "🚖",
                       initial_sidebar_state = 'expanded')
                   
    st.title("NYC Yellow Taxi Trip 🚖")
    #Selecting the type of prediction to perform
    pred_type = st.selectbox("What do you want to predict?",["trip_duration","taxi_fare"])

    if pred_type == "trip_duration":
        trip_duration_form()

    elif pred_type == "taxi_fare":
        trip_fare_form()

   
if __name__ == "__main__":
    main()



