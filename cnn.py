import pandas as pd
from keras.applications import VGG16
import tensorflow
import cv2
import sklearn
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
from io import StringIO
from keras.layers import Flatten
from keras.models import Model

x=vgg16.output
x=Flatten()(x)
Model=Model(inputs=vgg16.input,outputs=x)
Model.compile(optimizer="Adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

st.title("Dog Breed Classification")
uploaded_file = st.file_uploader("Enter an image of a dog")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img1 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img1 is None:
        st.error("Image not loaded correctly. Please check the file format.")
    else:
        st.image(img1, channels="BGR")
        if st.button("OK"):
            current_dir = os.path.dirname(__file__)
            pickle_file_path = os.path.join(current_dir, "random_forest.pkl")
            with open(pickle_file_path, "rb") as f:
                model = pickle.load(f)

            img2=cv2.resize(img1,(224,224))
            predictions = Model.predict(img2[np.newaxis])

            st.subheader(model.predict(predictions)[0])




        

           







