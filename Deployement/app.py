from flask import Flask		   # import flask
from flask import render_template
from predictions import get_feature_1
from flask import request
from werkzeug.utils import secure_filename
import os
import time
import cv2
from flask import Flask, render_template, request, redirect, flash, url_for, make_response

from detectron import detectron
from facedetector import facedetector

app = Flask(__name__)
app.secret_key = "secret key"
app.DetectronObj = detectron()

app.config['UPLOAD_FOLDER'] = './uploads'

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

            print(filename)

            start_time = time.time()
            image = get_feature_1(app.DetectronObj, app.config['UPLOAD_FOLDER']+'/'+filename)
            end_time = time.time()
            print('Time of execution: %s seconds' % str(end_time-start_time))
            retval, buffer = cv2.imencode('.png', image)
            response = make_response(buffer.tobytes())
            response.headers['Content-Type'] = 'image/png'

            return response

app.run(debug=True)
