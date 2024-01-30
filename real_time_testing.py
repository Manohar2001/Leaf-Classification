#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import tensorflow as tf
import numpy as np
import threading


# In[2]:


classes = ['Gauva (P3)-diseased', 'Gauva (P3)-healthy', 'Lemon (P10)-diseased','Lemon (P10)-healthy', 'Mango (P0)-diseased', 'Mango (P0)-healthy', 'Pomegranate (P9)-diseased', 'Pomegranate (P9)-healthy']



# In[3]:


model = tf.keras.models.load_model('plants-99.59.h5')


# In[4]:


def classify_frame(main_frame, classes):
    frame = cv2.resize(main_frame, (300, 200))
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    prediction = model.predict(frame)
    predicted_class = np.argmax(prediction, axis=1)
    class_label = classes[predicted_class[0]]
    frame = cv2.putText(cv2.resize(main_frame, (600, 400)), class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return frame, class_label

def process_video():
    ip_cam_url = "http://192.168.180.53:8080/video"
    cap = cv2.VideoCapture(ip_cam_url)
    if not cap.isOpened():
        print("Error: Could not open IP webcam.")
        return 
    else:
        print("Connected to IP webcam.")
    while True:
        ret, main_frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame, class_label = classify_frame(main_frame, classes)
        print(class_label)

        cv2.imshow('Leaf Classification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_thread = threading.Thread(target=process_video)
video_thread.start()
video_thread.join()


# In[ ]:




