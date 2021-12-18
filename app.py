from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
def index():
    if(request.method == 'GET'):
        print("GET istegi calisti!")
        return render_template('index.html',result = '')
    elif(request.method == 'POST'):
        print(request.files)
        print(request.files['file'])
        print(request.files['file'].filename)
        return render_template('index.html',result = request.files['file'])

app.run(host='0.0.0.0', port=5000)