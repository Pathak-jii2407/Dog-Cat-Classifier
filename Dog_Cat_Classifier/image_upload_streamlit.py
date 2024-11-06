import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os

st.title("Image Upload and Display")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img_path = Image.open(uploaded_file)

    # # Load the model from the same directory
    # model_filename = "dogVScat.h5"
    # model_path = os.path.join(os.getcwd(), model_filename)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "dogVScat.h5")

    # Load the model
    model = load_model(model_path)

    # Resize and preprocess the image
    image = load_img(uploaded_file, target_size=(150, 150))  # resize the image
    img_array = img_to_array(image)  # converting to array
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    img_array /= 255.0  # rescale

    # Make predictions
    predictions = model.predict(img_array)

    # Display the image and prediction result
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image)

    if predictions[0][0] > 0.5:
        ax.set_title(f'Predicted: Dog, Chances: {predictions[0][0] * 100:.2f}%')
    else:
        ax.set_title(f'Predicted: Cat, Chances: {(1 - predictions[0][0]) * 100:.2f}%')

    ax.axis('off')

    st.pyplot(img_array)
else:
    st.write("Please upload an image.")
