import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('house.csv')

dshouse = load_data()

# Train model function
def train_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    regresi = LinearRegression()
    regresi.fit(x_train, y_train)
    return regresi, x_test, y_test

# Sidebar user inputs
st.sidebar.title("House Price Prediction")
st.sidebar.write("Input house features to predict the price.")

# Input fields for house features
square_footage = st.sidebar.slider('Square Footage', float(dshouse['Square_Footage'].min()), float(dshouse['Square_Footage'].max()), 1500.0)
num_bedrooms = st.sidebar.slider('Number of Bedrooms', float(dshouse['Num_Bedrooms'].min()), float(dshouse['Num_Bedrooms'].max()), 3.0)
num_bathrooms = st.sidebar.slider('Number of Bathrooms', float(dshouse['Num_Bathrooms'].min()), float(dshouse['Num_Bathrooms'].max()), 2.0)
year_built = st.sidebar.slider('Year Built', float(dshouse['Year_Built'].min()), float(dshouse['Year_Built'].max()), 2000.0)
lot_size = st.sidebar.slider('Lot Size', float(dshouse['Lot_Size'].min()), float(dshouse['Lot_Size'].max()), 5000.0)
garage_size = st.sidebar.slider('Garage Size', float(dshouse['Garage_Size'].min()), float(dshouse['Garage_Size'].max()), 1.0)
neighborhood_quality = st.sidebar.slider('Neighborhood Quality', float(dshouse['Neighborhood_Quality'].min()), float(dshouse['Neighborhood_Quality'].max()), 7.0)

# Main section
st.title("House Price Prediction App")
st.write("### House Price Dataset Preview:")
st.dataframe(dshouse.head())

# Prepare data for model
x = dshouse[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']].values
y = dshouse['House_Price'].values

# Train the model
regresi, x_test, y_test = train_model(x, y)

# Predict based on user inputs
input_data = pd.DataFrame({
    'Square_Footage': [square_footage],
    'Num_Bedrooms': [num_bedrooms],
    'Num_Bathrooms': [num_bathrooms],
    'Year_Built': [year_built],
    'Lot_Size': [lot_size],
    'Garage_Size': [garage_size],
    'Neighborhood_Quality': [neighborhood_quality]
})

predicted_price = regresi.predict(input_data)

# Convert predicted price from USD to Rupiah
conversion_rate = 15000  # Example conversion rate, 1 USD = 15,000 IDR
predicted_price_rupiah = predicted_price[0] * conversion_rate

# Display the predicted price in Rupiah
st.write(f"### Predicted House Price: Rp {predicted_price_rupiah:,.2f}")

# Show correlation and regression plots
st.write("### Regression Plots:")
independent_columns = dshouse.columns[dshouse.columns != 'House_Price']
for column in independent_columns:
    plt.figure(figsize=(6, 4))
    sns.regplot(x=dshouse[column], y=dshouse['House_Price'], scatter_kws={'alpha': 0.5}, line_kws={"color": "red"})
    plt.xlabel(column)
    plt.ylabel('House Price')
    plt.title(f'Regression: {column} vs House Price')
    st.pyplot(plt.gcf())  # Display the plot in the Streamlit app

# Model evaluation
st.write("### Model Evaluation:")
y_pred = regresi.predict(x_test)
r2 = regresi.score(x_test, y_test)
st.write(f"R-squared value of the model: {r2:.2f}")