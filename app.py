from flask import Flask,render_template,request,send_file
import base64
from PIL import Image
import io
import cv2
import numpy as np

app = Flask(__name__,
            static_url_path='', 
            static_folder='web/static',
            template_folder='web/templates')

@app.route('/',methods = ['GET','POST'])
def index():
    if(request.method == 'GET'):
        return render_template('index.html',result = '')

@app.route('/video',methods = ['GET','POST'])
def video():
    if(request.method == 'GET'):
        return render_template('video.html',result = '')

@app.route('/proc-image',methods = ['POST'])
def procImage():
    if(request.method == 'POST'):
        imageString = request.get_json(force=True)['imageString']
        img = convertBase64ToImage(imageString)
        img = np.array(img)
        #todo input to ml model
        retval, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer)
        return jpg_as_text

def convertImageToBase64(image):
    encoded_string = base64.b64encode(image.read())
    return encoded_string

def convertBase64ToImage(base64Code):
    imageString = base64Code.split(',')[1]
    return Image.open(io.BytesIO(base64.b64decode(bytes(imageString, "utf-8"))))

app.run(host='0.0.0.0', port=3000)