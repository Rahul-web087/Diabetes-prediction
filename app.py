import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image

import pickle

with open("content/trained_model.sav", "rb") as f:
    model = pickle.load(f)


# create the Streamlit app
def app():

    # img = Image.open(r"img.jpeg")
    # img = img.resize((200,200))
    # st.image(img,caption="Diabetes Image",width=200)


    st.title('Diabetes Prediction')

    # create the input form for the user to input new data
    st.sidebar.title('Input Features')
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # make a prediction based on the user input
    input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
    input_data_nparray = np.asarray(input_data)
    reshaped_input_data = input_data_nparray.reshape(1, -1)
    prediction = model.predict(reshaped_input_data)

    # display the prediction to the user
    st.write('Based on the input features, the model predicts:')
    if prediction == 1:
        st.warning('This person has diabetes.')
    else:
        st.success('This person does not have diabetes.')

    # display some summary statistics about the dataset
   # st.header('Dataset Summary')
   # st.write(diabetes_df.describe())

  #  st.header('Distribution by Outcome')
   # st.write(diabetes_mean_df)

    # display the model accuracy
   # st.header('Model Accuracy')
    # st.write(f'Train set accuracy: {train_acc:.2f}')
    # st.write(f'Test set accuracy: {test_acc:.2f}')

if __name__ == '__main__':
    app()