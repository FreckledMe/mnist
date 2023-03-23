import streamlit as st
import cv2
from tensorflow import keras
import numpy as np

st.title('Rasmga qarab sonni topib beraman')


model = keras.models.load_model(r'C:\Users\justf\Notebooks\mnist\model\mnist_model.h5')
image = st.file_uploader(label='Image upload',type=['png','jpg'])
if image:
    # img = image.getvalue()
    img = cv2.imread(image.name)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img,dsize=(28,28))
    gray_img = gray_img.astype('float32')
    gray_img /= 255
    gray_img = np.array([gray_img])


with st.form("key1"):
    # ask for input
    button_check = st.form_submit_button("Predict")
if button_check:
    st.text(f'Bu raqam {np.argmax(model.predict(gray_img))}')