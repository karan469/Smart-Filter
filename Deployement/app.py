from flask import Flask		   # import flask
from flask import render_template
from predictions import getPrediction
from flask import request
from werkzeug.utils import secure_filename
import os
import cv2
from flask import Flask, render_template, request, redirect, flash, url_for, make_response


app = Flask(__name__)
app.secret_key = "secret key"

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
            # getPrediction(filename)
            # label, acc = getPrediction(filename)
            print(filename)
            image = getPrediction(app.config['UPLOAD_FOLDER']+'/'+filename)
            retval, buffer = cv2.imencode('.png', image)
            response = make_response(buffer.tobytes())
            response.headers['Content-Type'] = 'image/png'
            return response
            # flash(label)
            # flash(acc)
            # flash(filename)
            # return redirect('/')

app.run(debug=True)
