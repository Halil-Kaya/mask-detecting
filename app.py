from flask import Flask,Response,render_template,request,send_file
import base64
from PIL import Image
import io
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

vc = cv2.VideoCapture(0)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False


CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4

loaded_model = keras.models.load_model("yolo/iv3_mask-model.h5")
mask_types = ["un-mask","mask","improper-mask"]
model_weights = "yolo/yolov4-obj_last.weights"
model_config = "yolo/yolov4-obj.cfg"
net = cv2.dnn.readNet(model_weights, model_config)
colors =  [(255,255,0),(0,255,255),(255,0,255)]

app = Flask(__name__,
            static_url_path='', 
            static_folder='web/static',
            template_folder='web/templates')

@app.route('/',methods = ['GET','POST'])
def index():
    if(request.method == 'GET'):
        return render_template('index.html',result = '')

@app.route('/video-page',methods = ['GET','POST'])
def videoPage():
    if(request.method == 'GET'):
        return render_template('video-page.html',result = '')

@app.route('/proc-image',methods = ['POST'])
def procImage():
    if(request.method == 'POST'):
        imageString = request.get_json(force=True)['imageString']
        orginalImage = convertBase64ToImage(imageString)
        orginalImage = np.array(orginalImage)
        
        img = run_yolo_frame(orginalImage,net,loaded_model)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        retval, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer)
        return jpg_as_text

@app.route('/proc-video',methods = ['POST'])
def procVideo():
    if(request.method == 'POST'):
        video = request.files['video']
        video_path = video.filename
        video.save(os.path.join('', video_path))
        cap = cv2.VideoCapture(video_path)
        if(cap.isOpened()==False):
            print("Error openning video stream or file")
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        size = (frame_width, frame_height)

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('./web/static/output.avi', fourcc, 20.0, size)
        
        myFrameNumber = 10
        # check for valid frame number
        if myFrameNumber >= 0 & myFrameNumber <= totalFrames:
            # set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)

        while True:
            success, frame = cap.read()
            if success:
                frame = cv2.resize(frame, size)
                frame = run_yolo_frame(frame,net,loaded_model)
                # write processed data
                out.write(frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        # After the loop release the cap object
        
        cap.release()
        cv2.destroyAllWindows()
        out.release()
        return "OK"

def gen(vc):
    while True:
        rval, frame = vc.read()
        print(frame)
        #todo: put the img to ml model
        #frame = run_yolo_frame(frame,net,loaded_model)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    global vc
    return Response(gen(vc),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def convertVideoToBase64(video):
    encoded_string = base64.b64encode(video.read())
    return encoded_string

def convertBase64ToImage(base64Code):
    imageString = base64Code.split(',')[1]
    return Image.open(io.BytesIO(base64.b64decode(bytes(imageString, "utf-8"))))


def run_yolo_frame(img,net,mask_model):
    height, width, channels = img.shape
    print("height :",height)
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
                print("%%%%")
                print(confidence)
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
            print(percent)
            confidence = str(round(percent,2))
            crop_img = img[y:y + h, x:x + w]

            try:
                mask_model_input = cv2.resize(crop_img,(256,256))
                mask_model_input = np.expand_dims(mask_model_input, axis=0)

                result = mask_model.predict(mask_model_input)
                print(result)
                chosen = np.argmax(result)
                color = colors[chosen]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
                mask_confidence = str(round(result[0][chosen],3))
                class_name = mask_types[np.argmax(result)]
                label = str(class_name)
                print("LABEL : "+label)
                #label = "face %"
                cv2.putText(img, label + " " + mask_confidence, (x, y-5),
                             font, fontScale=1, color=(255, 255, 255),
                             thickness=2, lineType=cv2.LINE_AA)

            except:
                print("HATA")

    return img

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, threaded=True,debug=True)
