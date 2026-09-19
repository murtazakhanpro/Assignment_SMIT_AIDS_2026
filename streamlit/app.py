import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Model load
model = tf.keras.models.load_model("veg_model.keras")

# Classes
class_names = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage',
    'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn',
    'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes',
    'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango',
    'onion', 'orange', 'paprika', 'pear', 'peas',
    'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans',
    'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip',
    'watermelon'
]

# UI
st.title("🍎 Fruit & Vegetable Classifier")
st.write("Upload an image and get the prediction.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "jfif"]
)

if uploaded_file is not None:

    # Image load
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image")

    # Resize
    image = image.resize((180, 180))

    # Convert to array
    img_array = np.array(image)

    # Batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Softmax
    score = tf.nn.softmax(prediction[0])

    # Class
    predicted_class = class_names[np.argmax(score)]

    # Confidence
    confidence = np.max(score) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    # start with some new