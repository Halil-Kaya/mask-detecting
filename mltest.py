import cv2
import pickle
import numpy as np

loaded_model = pickle.load(open("pickle_model.sav", 'rb'))

path = r"static/bg.jpeg"
img = cv2.imread(path)
img = cv2.resize(img,(256,256))
image = np.expand_dims(img, axis=0)
result = loaded_model.predict(image)
print("sonucu ekrana bastiriyorum : ")
print(result)