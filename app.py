import streamlit as st
from joblib import load
# Load the required libraries and objects
import numpy as np


# Load your trained model and vectorizer

# Load your trained model
model = load('precily.joblib')
vectorizer = load('vectorizer.joblib')

# Define the predict_similarity function
def predict_similarity(text1, text2):
    # Convert the input texts into numerical representations using TF-IDF
    text1_vector = vectorizer.transform([text1])
    text2_vector = vectorizer.transform([text2])

    # Concatenate the text vectors to create the input features for the model
    X = np.hstack([text1_vector.toarray(), text2_vector.toarray()])

    # Use the trained model to predict the similarity score between the input texts
    similarity_score = model.predict(X)
    return similarity_score[0]





# Create two text input fields in the Streamlit app
text1 = st.text_input("Enter first text")
text2 = st.text_input("Enter second text")

# When both texts are entered, send them to the model and display the similarity score
# Use the predict_similarity function to calculate the similarity between two texts
# similarity_score = predict_similarity(text1, text2)
if text1 and text2:
    similarity_score = predict_similarity(text1, text2)
    st.write(f"Similarity score: {similarity_score}")
