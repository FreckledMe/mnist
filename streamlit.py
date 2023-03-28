import streamlit as st
import cv2
from tensorflow import keras
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import numpy as np

st.title('Number prediction')
key = st.selectbox('Choose image type',('Draw by hand','Upload image'))
model = keras.models.load_model('model/mnist_model.h5')
class_names = ['0','1','2','3','4','5','6','7','8','9']

choose = st.sidebar.selectbox(
    "Classes:", class_names)
if choose:
    class_img = Image.open('class_images/'+choose+'.png')
    gray_img = ImageOps.grayscale(class_img)
    st.sidebar.image(gray_img)

if key == 'Upload image':
    uploaded_image = st.file_uploader(label='Image upload',type=['png','jpg'])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()),dtype=np.uint8)
        cv_image = cv2.imdecode(file_bytes,1)
        gray_img = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img,dsize=(28,28))
        gray_img = gray_img.astype('float32')
        gray_img /= 255
        gray_img = np.array([gray_img])
        for_display_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        for_display_image = cv2.resize(for_display_image,dsize=(120,120,))
        st.image(for_display_image)

    with st.form("key1"):
        # ask for input
        button_check = st.form_submit_button("Predict")
    if button_check:
        st.text(f'Bu raqam {np.argmax(model.predict(gray_img))}')
else:
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 10)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color='#FFFFFF',
        background_color='#000000',
        update_streamlit=True,
        width = 250,
        height=250,
        drawing_mode='freedraw',
        point_display_radius= 0,
        key="canvas",
    )

    if st.button(label='Predict') and canvas_result.image_data is not None:
        img = canvas_result.image_data
        img = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
        img = cv2.resize(img,(28,28))
        img = img.astype('float32')
        img /= 255
        img = np.array([img])
        
        st.text(class_names[np.argmax(model.predict(img))])