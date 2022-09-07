import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import keras
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

def streamlit():
    
    classifier_model = 'model_v5.h5'
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    
    labels = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']
    try:
        file = st.sidebar.file_uploader(label = 'Upload an image', type = ["png","jpg","jpeg"] )
        class_btn = st.sidebar.button("Classify")
        
        
        
        if file is not None: 
                
            image = Image.open(file)
            st.image(image, caption='Uploaded Image', use_column_width=None)
            
            basewidth = 28
            img = Image.open(file).convert('L')
            width_percent = (basewidth / float(img.size[0]))
            hight_size = int((float(img.size[1]) * float(width_percent)))
            img = img.resize((basewidth, hight_size), Image.ANTIALIAS)

            img_np = np.array(img) / 255 # noramlize
            img_np.reshape(1, 70, 70, 3)
            img_np = np.array([img_np]) # add dimesion
            
            if class_btn == True:
                prediction = model.predict(img_np)
                pred = prediction[0][0]
                st.write("Prediction of image is :")
                st.write(labels[pred])
                st.success('Klasifikasi')
        if file is  None: 
                st.title('Deteksi Objek dengan Model CNN')

                st.subheader("Welcome to this simple web application. Klasifikasi objek dalam 8 kelas yang berbeda yaitu: Pesawat, Mobil, Kucing, Anjing, Bunga, Buah-buahan, Motor, Manusia")
        else:
            pass        
    except:
        
        st.warning("Tolong Upload Gambar Yang Bener")
        st.info("Refresh")
   
streamlit()
