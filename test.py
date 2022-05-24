import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

loaded_model = keras.models.load_model("yolo/iv3_mask-model.h5")
mask_types = ["un-mask","mask","improper-mask"]
model_weights = "yolo/yolov4-obj_last.weights"
model_config = "yolo/yolov4-obj.cfg"
net = cv2.dnn.readNet(model_weights, model_config)
colors =  [(255,255,0),(0,255,255),(255,0,255)]

def run_yolo_frame(img,net,mask_model):
    height, width, channels = img.shape
    #print("height :",height)
    # blob from Image returns the input image after doing mean substraction, normalizing, and channels wrapping
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    # blob_result(blob,img)

    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()  # to get the output layer names
    layerOutputs = net.forward(output_layers_names)  # get output from this function

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        # second loop is to extract infromation from each output
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                #print("%%%%")
                #print(confidence)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #print(len(boxes))
    # Filter box used non maximum supressions
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
                               NMS_THRESHOLD)  # remove redundant(unnecessary) boxes
    #print(indexes.flatten())
    font = cv2.FONT_HERSHEY_COMPLEX
    
    if len(indexes) > 0: # to prevent AttributeError: 'tuple' object has no attribute 'flatten'
        for i in indexes.flatten():
            x, y, w, h = boxes[i]

            label = str(mask_types[class_ids[i]])
            percent = confidences[i]*100
            #print(percent)
            confidence = str(round(percent,2))
            
            crop_img = img[y:y + h, x:x + w]

            try:
                mask_model_input = cv2.resize(crop_img,(256,256))
                mask_model_input = np.expand_dims(mask_model_input, axis=0)

                result = mask_model.predict(mask_model_input)
                #print(result)
                chosen = np.argmax(result)
                # 
                color = colors[chosen]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
                #print(chosen)
                mask_confidence = str(round(result[0][chosen],3))
                class_name = mask_types[np.argmax(result)]
                label = str(class_name)
                #print("LABEL : "+label)
                #label = "face %"
                cv2.putText(img, label + " " + mask_confidence, (x, y-5),
                             font, fontScale=1, color=(255, 255, 255),
                             thickness=2, lineType=cv2.LINE_AA)

            except:
                print("HATA")

    return img
"""
video_path = "Kumru.mp4"
cap = cv2.VideoCapture(video_path)
if(cap.isOpened()==False):
    print("Error openning video stream or file")
totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("TOTAL FRAME COUNT :",totalFrames)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('Result.mp4', fourcc, 20.0, size)

myFrameNumber = 10
# check for valid frame number
if myFrameNumber >= 0 & myFrameNumber <= totalFrames:
    # set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)
count = 0
while True:
    success, frame = cap.read()
    if success == True:
        
        frame = cv2.resize(frame, size)
        
        frame = run_yolo_frame(frame,net,loaded_model)
        #cv2.putText(frame,f'FPS : {int(myFrameNumber)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),2)
        count+=1
        out.write(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break
# After the loop release the cap object
print("Video is finished")
cap.release()
cv2.destroyAllWindows()
out.release()
"""

"""
vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)
frame_rate = 0.005
prev = 0
while (True):
    stime = time.time() # start time
    ctime = stime-prev # time elapse
    print("TIME : ",ctime)
    ret, frame = vid.read()
    if frame is not None:
        frame = cv2.flip(frame, 1)
        if ctime > frame_rate:
            frame = run_yolo_frame(frame,net,loaded_model)
        prev = time.time()# change prev
        cv2.imshow('frame', frame)
    else:
        print("There is no image")


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows


cv2.destroyAllWindows()
"""
import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    frame = run_yolo_frame(frame,net,loaded_model)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")