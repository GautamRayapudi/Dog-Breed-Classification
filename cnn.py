import pandas as pd
from keras.applications import VGG16
import tensorflow as tf
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st
from keras.layers import Flatten
from keras.models import Model

# Load VGG16 model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = vgg16.output
x = Flatten()(x)
model1 = Model(inputs=vgg16.input, outputs=x)
model1.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

st.title("Dog Breed Classification")
uploaded_file = st.file_uploader("Enter an image of a dog")

if uploaded_file is not None:
    def display_feature_maps(model, img):
        layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
        activation_model = Model(inputs=model.input, outputs=layer_outputs)
        img = img[np.newaxis]
        feature_maps = activation_model.predict(img)

        for layer_name, feature_map in zip([layer.name for layer in model.layers if 'conv' in layer.name], feature_maps):
            n_features = feature_map.shape[-1]
            cols = 8
            rows = n_features // cols
            fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
            for i in range(n_features):
                ax = axes[i // cols, i % cols]
                ax.matshow(feature_map[0, :, :, i],cmap="gray")
                ax.axis('off')
            st.write(f"Feature maps for layer: {layer_name}")
            st.pyplot(fig)

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img1 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img1 is None:
        st.error("Image not loaded correctly. Please check the file format.")
    else:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        st.image(img1_rgb, channels="RGB")
        
        if st.button("OK"):
            current_dir = os.path.dirname(__file__)
            pickle_file_path = os.path.join(current_dir, "random_forest.pkl")
            with open(pickle_file_path, "rb") as f:
                model = pickle.load(f)

            img2 = cv2.resize(img1_rgb, (224, 224))

            predictions = model1.predict(img2[np.newaxis])

            st.subheader(model.predict(predictions)[0])

            display_feature_maps(model1, img2)
