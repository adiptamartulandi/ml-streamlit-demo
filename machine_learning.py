#this apps will be used to predict iris dataset
import streamlit as st
import pandas as pd
import numpy as np
import joblib


#title
st.title("Iris Flower Prediction")

#load model
model = joblib.load("iris_model.joblib")

#input form numeric
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

#upload files
# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)

#input data katregoriukal
# species = st.selectbox("Species", ["setosa", "versicolor", "virginica"])

# Data Preprocessing
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

#encode species
# if species == "setosa":
#     input_data = np.append(input_data, [[1, 0, 0]], axis=0)
# elif species == "versicolor":
#     input_data = np.append(input_data, [[0, 1, 0]], axis=0)
# else:
#     input_data = np.append(input_data, [[0, 0, 1]], axis=0)

#press button to predict
# if st.button("Predict"):
#     list_predictoon = []
#     for data in df:
#         input_data = np.array([data])
#         prediction = model.predict(input_data)
#         st.write(prediction)
#         list_predictoon.append(prediction[0])
#         st.write(["setosa", "versicolor", "virginica"][prediction[0]])

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(prediction)
    st.write(["setosa", "versicolor", "virginica"][prediction[0]])