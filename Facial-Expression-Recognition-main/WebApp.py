import streamlit as st
import numpy as np
import tensorflow as tf
import h5py
from tensorflow.keras.models import load_model
# import cv2 as cv
from PIL import Image, ImageOps


vgg16 = load_model('best_model.h5')

st.set_page_config(page_title='Facial Expression Recognition', layout='wide')
st.header("""how you doin'?""")
# st.footer("Project by Abhishek Biswas")
st.write("This is a Machine Learning Model Trained to recognise Facial Expressions")

file = st.file_uploader("Upload an Image", type = ['jpg', 'png'])

st.write("OR")

def pred(img):
    size = (48, 48,)
    img = ImageOps.fit(img, size, Image.ANTIALIAS)
    img = ImageOps.grayscale(img)
    img = np.array(img)
    img = img/255.
    img_reshape = img[np.newaxis, ...]
    p = vgg16.predict(img_reshape)

    return p

if file is not None:
    img = Image.open(file)
    st.image(img, width = 300 )
    p = pred(img)
    class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    s = "This Image is Most Likely a : "+class_names[np.argmax(p)]
    st.success(s)

#if st.button("camera"):
 #   file1 = st.camera_input("")
#     file = file.get_value()
#     file = cv.imdecode(np.frombuffer(bytes_data, np.uint8), cv.IMREAD_COLOR)
  #  if file1 is not None:
   #     img = Image.open(file1)
    #    st.image(img, width = 300 )
     #   p = pred(img)
      #  class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
       # s = "This Image is Most Likely a : "+class_names[np.argmax(p)]
        #st.success(s)
    
