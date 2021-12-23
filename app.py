from flask import Flask,render_template,request,send_file
import cv2
import pickle
import numpy as np

loaded_model = pickle.load(open("pickle_model.sav", 'rb'))

app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
def index():
    if(request.method == 'GET'):
        return render_template('index.html',result = '')


@app.route('/find',methods = ['POST'])
def find():
    if(request.method == 'POST'):
        file = request.files['file']
        npimg = np.fromfile(file,np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = cv2.resize(img,(256,256))
        image = np.expand_dims(img, axis=0)
        result = loaded_model.predict(image)
        print(result[0])
        resultString = "maskesiz olma olasiligi : " + str(result[0][0]) + " maskeli olmama olasiligi : " + str(result[0][1]) + " maske hatali olma olasiligi : " + str(result[0][2])
        response = {}
        response['result'] = resultString
        return response


app.run(host='0.0.0.0', port=3000)