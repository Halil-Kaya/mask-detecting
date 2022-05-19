from flask import Flask,Response,render_template,request,send_file
import base64
from PIL import Image
import io
import cv2
import numpy as np

app = Flask(__name__,
            static_url_path='', 
            static_folder='web/static',
            template_folder='web/templates')

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.samples.findFile("./haarcascade_frontalface_alt.xml"))

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
        img = convertBase64ToImage(imageString)
        img = np.array(img)
        #todo: put the img to ml model

        retval, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer)
        return jpg_as_text

@app.route('/proc-video',methods = ['POST'])
def procVideo():
    if(request.method == 'POST'):
        videoString = request.get_json(force=True)['videoString']
        video = convertBase64ToVideo(videoString)
        #img = np.array(img)
        return "test"

def gen(video):
    while True:
        success, image = video.read()
        #todo: put the img to ml model

        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = face_cascade.detectMultiScale(frame_gray)

        for (x, y, w, h) in faces:
            center = (x + w//2, y + h//2)
            cv2.putText(image, "X: " + str(center[0]) + " Y: " + str(center[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            faceROI = frame_gray[y:y+h, x:x+w]
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
    pass

def convertBase64ToVideo(base64Code):
    videoString = base64Code.split(',')[1]
    return io.BytesIO(base64.b64decode(bytes(videoString, "utf-8")))

def convertImageToBase64(image):
    encoded_string = base64.b64encode(image.read())
    return encoded_string

def convertBase64ToImage(base64Code):
    imageString = base64Code.split(',')[1]
    return Image.open(io.BytesIO(base64.b64decode(bytes(imageString, "utf-8"))))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, threaded=True)