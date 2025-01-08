import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained models
model = joblib.load('/Users/k_krishna/Downloads/IDS_Project/multioutput_stacking_regressor.pkl')
preprocessor = joblib.load('preprocessing_pipeline.pkl')

# Streamlit application layout
st.title('Crime Rate Prediction Model')

# Define your county list here
counties = ["Allegany County", "Anne Arundel County", "Baltimore City", "Baltimore County", 
            "Calvert County", "Caroline County", "Carroll County", "Cecil County", "Charles County", 
            "Dorchester County", "Frederick County", "Garrett County", "Harford County", "Howard County", 
            "Kent County", "Montgomery County", "Prince George's County", "Queen Anne's County", 
            "Somerset County", "St. Mary's County", "Talbot County", "Washington County", 
            "Wicomico County", "Worcester County"]

# Create input fields for the features
data_input = {
    'County': st.selectbox('County', counties),
    'Year': 2025,  # Fixed year
    'Grades Pre-K': st.slider('Grades Pre-K', min_value=0, max_value=1000, value=500, step=10),
    'Grades K-5': st.slider('Grades K-5', min_value=0, max_value=5000, value=2500, step=10),
    'Grades 6-8': st.slider('Grades 6-8', min_value=0, max_value=3000, value=1500, step=10),
    'Grades 9-12': st.slider('Grades 9-12', min_value=0, max_value=4000, value=2000, step=10),
    'Unemploy_Value': st.slider('Unemployment Value', min_value=0, max_value=50000, value=25000, step=100),
    'POPULATION': st.slider('Population', min_value=0, max_value=2000000, value=1000000, step=1000),
    'B & E': st.slider('B & E', min_value=0, max_value=1000, value=500, step=10),
    'LARCENY THEFT': st.slider('Larceny Theft', min_value=0, max_value=5000, value=2500, step=10),
    'M/V THEFT': st.slider('M/V Theft', min_value=0, max_value=1500, value=750, step=10),
    'GRAND TOTAL': st.slider('Grand Total', min_value=0, max_value=10000, value=5000, step=10),
    'PROPERTY CRIME TOTALS': st.slider('Property Crime Totals', min_value=0, max_value=10000, value=5000, step=10)
}

# Button to make prediction
if st.button('Predict'):
    input_df = pd.DataFrame([data_input])
    input_processed = preprocessor.transform(input_df)
    predictions = model.predict(input_processed)
    
    # Ensure the predictions are handled as expected for multiple outputs
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        crime_types = ['MURDER', 'RAPE', 'ROBBERY', 'AGG. ASSAULT', 'VIOLENT CRIME TOTAL']
        plt.figure(figsize=(10, 5))
        plt.bar(crime_types, predictions[0], color='blue', alpha=0.7)
        plt.title('Predicted Crime Rates')
        plt.xlabel('Crime Types')
        plt.ylabel('Predicted Rates')
        st.pyplot(plt)
    else:
        st.write(f'Predicted Crimes: {predictions[0]:.2f}')

# To run the Streamlit app, save this script as app.py and run using `streamlit run app.py` from your command line.
