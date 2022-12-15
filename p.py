import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def visualize_images(img):
    "Preserves the scale of the image"  
    imgs = [img]
    imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
    fig, (ax1) = plt.subplots(1)
    ax1.imshow(imgs[0])
    ax1.axis('off')
    st.pyplot(fig)

def main():
    st.title("Emotion Recognition")
    st.markdown('Build with Streamlit')    
    choice = st.sidebar.radio("Select a Method:", ("Face Recognition", "Etc"))
    if choice == "Face Recognition":       
        st.subheader("Face Recognition")        
        image_file = st.file_uploader(
            "Upload the first image", type=['jpeg', 'png', 'jpg'])             
        if image_file:           
            image = cv2.imread(image_file.name)          
            if st.button("Process"):              
                result = DeepFace.analyze(image, actions = ["age", "gender", "emotion", "race"])
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, 1.1, 4)
                for (x,y,w,h) in faces:
                    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image,
                     result['dominant_emotion'],
                     (10, 100),
                     font, 3,
                     (0,0,255),
                     3,
                     cv2.LINE_4);
                visualize_images(image)
                st.json(json.dumps(result, indent=4))
    if choice == "Etc":    
        st.subheader("Etc")
if __name__ == "__main__":
    main()