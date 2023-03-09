import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('hk.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
regressor=data["model"]
BREED = data["BREED"]
RIGHT_EYE_SIZE = data["RIGHT_EYE_SIZE"]



def show_predict_page():
    st.title("DOG AGE PREDICTION")

    st.write("""### We need some information to predict the AGE""")
    
        
    BREED = ("German shepherd","Rottweiler","Saint Bernard","Doberman","Lab","Golden retriever","Great dane","Husky","Dalmatian","Cane corso","Beagle","Poodle","Pitbull","Sheltie","Bassett hound","Cocker spaniel","Australian shepherd","English spinger spaniel","Adghan hound","Pug","Daschund","Pomeranian","Chihuahua","Shihtzu","Maltese")

    BREED = st.selectbox("BREED", BREED)
    
    #education = st.selectbox("Education Level", education)

    RIGHT_EYE_SIZE = st.slider("EYE_SIZE", 0, 20, 1)

    ok = st.button("Predict AGE")
    if ok:
        X = np.array([[BREED,RIGHT_EYE_SIZE ]])
        #X = X.astype(float)

        AGE = regressor.predict(X)
        st.subheader(f"The estimated AGE is ${AGE[0]:.2f}")
        
     
