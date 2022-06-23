import os
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.utils import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf

import random
import time

# load model
model = load_model("C:\Python310\File\checkpoint_cow.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture("C:/Python310/File/video_test.mp4")

while True:
    ret, test_img = cap.read()
    
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5) 

    for (x, y, w, h) in faces_detected:
        
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = keras.utils.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        max_index = np.argmax(predictions[0])

        animal = ('Cat', 'Chicken', 'Cow', 'Dog', 'Horse', 'Sheep') 
        predicted_animal = animal[max_index]

    cv2.putText(test_img, "So bo la: " + str(len(predicted_animal)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
        
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Animal analysis', resized_img)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows
