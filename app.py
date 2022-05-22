from flask import Flask,Response,render_template,request,send_file
import base64
from PIL import Image
import io
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

loaded_model = keras.models.load_model("iv3_mask-model.h5")
mask_types = ["un-mask","mask","improper-mask"]

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
        img = np.array(orginalImage)

        height, width, channels = img.shape
        img = cv2.resize(img,(256,256))
        mask_model_input = np.expand_dims(img,axis = 0)

        resultImage = loaded_model.predict(mask_model_input)
        chosen = np.argmax(resultImage)
        class_name = mask_types[chosen]
        label = str(class_name)

        orginalImage = cv2.cvtColor(orginalImage, cv2.COLOR_BGR2RGB)
        cv2.putText(orginalImage,label,(0,10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        
        retval, buffer = cv2.imencode('.jpg', orginalImage)
        jpg_as_text = base64.b64encode(buffer)
        return jpg_as_text

@app.route('/proc-video',methods = ['POST'])
def procVideo():
    if(request.method == 'POST'):
        video = request.files['video']
        video_path = video.filename
        video.save(os.path.join('', video_path))
        cap = cv2.VideoCapture(video_path)
        #TODO put in to model
        return "resultString"

def gen(video):
    while True:
        success, image = video.read()
        #todo: put the img to ml model

        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def convertVideoToBase64(video):
    encoded_string = base64.b64encode(video.read())
    return encoded_string

def convertBase64ToImage(base64Code):
    imageString = base64Code.split(',')[1]
    return Image.open(io.BytesIO(base64.b64decode(bytes(imageString, "utf-8"))))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, threaded=True,debug=True)
