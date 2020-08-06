from flask import Flask		   # import flask
from flask import render_template
from predictions import get_feature_1
from predictions import get_feature_2
from flask import request
from werkzeug.utils import secure_filename
import os
import time
import cv2
from flask import Flask, render_template, request, redirect, flash, url_for, make_response

from detectron import detectron
from facedetector import facedetector
from smile import smiledetector

app = Flask(__name__)
app.secret_key = "secret key"
app.DetectronObj = detectron()
# app.smile_detector = smiledetector('../../resnet50_smiledetection.h5')
app.smile_detector = smiledetector('../../smiledetection.h5')

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
        bg_file = None
        bg_file = request.files['bg-file']
        bg_filename = None

        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

            if(bg_file):
                bg_filename = secure_filename(bg_file.filename)
                bg_file.save(os.path.join(app.config['UPLOAD_FOLDER'], bg_filename))

            unsplash_params = request.form.getlist('params')[0]
            # TO-DO: Sanitize unsplash_params

            whichFunc = (request.form['options'])
            enableFaceFeatures = request.form['options-face-features']

            start_time = time.time()

            if(whichFunc=="without"):
                if(enableFaceFeatures=="disable-face-features"):
                    image = get_feature_2(app.DetectronObj, smile_detector=None, filename=app.config['UPLOAD_FOLDER']+'/'+filename, bg_filename=None, category=None)
                else:
                    image = get_feature_2(app.DetectronObj, smile_detector=app.smile_detector, filename=app.config['UPLOAD_FOLDER']+'/'+filename, bg_filename=None, category=None)

            elif(whichFunc=="custom"):
                if(bg_file==None or bg_filename==None):
                    flash('No background file selected for uploading')
                    return redirect(request.url)

                if(enableFaceFeatures=="enable-face-features"):
                    image = get_feature_2(app.DetectronObj, smile_detector=app.smile_detector, filename=app.config['UPLOAD_FOLDER']+'/'+filename, bg_filename=app.config['UPLOAD_FOLDER']+'/'+bg_filename, category=None)
                elif(enableFaceFeatures=="disable-face-features"):
                    image = get_feature_2(app.DetectronObj, smile_detector=None, filename=app.config['UPLOAD_FOLDER']+'/'+filename, bg_filename=app.config['UPLOAD_FOLDER']+'/'+bg_filename, category=None)
                else:
                    flash('Wrong operation at backend')
                    return redirect(request.url)

            elif(whichFunc=="unsplash"):
                if(unsplash_params==''):
                    unsplash_params='red'
                if(enableFaceFeatures=="enable-face-features"):
                    image = get_feature_2(app.DetectronObj, smile_detector=app.smile_detector, filename=app.config['UPLOAD_FOLDER']+'/'+filename, bg_filename=None, category=unsplash_params)
                elif(enableFaceFeatures=="disable-face-features"):
                    image = get_feature_2(app.DetectronObj, smile_detector=None, filename=app.config['UPLOAD_FOLDER']+'/'+filename, bg_filename=None, category=unsplash_params)
                else:
                    flash('Wrong operation at the backend.')
                    return redirect(request.url)

            end_time = time.time()
            if(image is None):
                flash('No results found.')
                return redirect(request.url)

            print('STATUS: Time of execution: %s seconds' % str((end_time-start_time)))

            retval, buffer = cv2.imencode('.png', image)
            response = make_response(buffer.tobytes())
            response.headers['Content-Type'] = 'image/png'

            return response

app.run(debug=True)
