import pickle

from flask import Flask
import dlib
import cv2
import pandas as pd
import numpy as np
from flask import url_for, redirect, render_template,  Response
from flask import Flask, request, render_template, url_for
from wtforms import StringField, RadioField, PasswordField, validators, FloatField, HiddenField
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from werkzeug.utils import secure_filename
from camera import Camera
from foundation import *
from lipstick import *


app = Flask(__name__)
app.config['SECRET_KEY'] = 'ea817c1b47d0976ca5031226da5e7229'

images=[]
names=[]



class UploadForm(FlaskForm):
    file = FileField('Upload the image here!')

class QuizForm1(FlaskForm):
    hf = HiddenField()

'''class MagicMirror(FlaskForm):
    lipstick=HiddenField()
    eyeshadow=HiddenField()
    blush=HiddenField()

class LipstickMatchForm(FlaskForm):
    file = FileField()

class FoundationMatchForm(FlaskForm):
    file= FileField()



def LipstickMatch(img):
    undertone= pipeline2()
    return lipstick'''

camera = Camera()

def gen(camera):
    while True:
        frame = camera.return_effect()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def start_streaming():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/lipstick_match', methods=['GET', 'POST'])
def lipstick_match():
    form =UploadForm()
    if form.validate_on_submit():
        filename = secure_filename(form.file.data.filename)
        form.file.data.save('/Users/aishwarya/PycharmProjects/flaskProject3/face' + filename)
        path='/Users/aishwarya/PycharmProjects/flaskProject3/face' + filename
        wrist = cv2.imread(path)
        undertone= pipeline2(wrist)
        skin =pipeline3(wrist)
        result=lipstick_reco(skin, undertone)

        for key in result.keys():
            names.append(key)
            images.append(result[key])
        print(undertone)
        print(lipstick_reco(skin, undertone))
        return render_template('lipstick.html', images=images, names=names)
    return render_template('upload.html', form=form,  attr="Lipstick")


@app.route('/foundation_match', methods=['GET','POST'])
def foundation_match():
    form = UploadForm()
    if form.validate_on_submit():
        filename = secure_filename(form.file.data.filename)
        form.file.data.save('/Users/aishwarya/PycharmProjects/flaskProject3/face' + filename)
        path = '/Users/aishwarya/PycharmProjects/flaskProject3/face' + filename
        wrist = cv2.imread(path)
        res=pipeline(wrist)
        images=[]
        names=[]
        for key in res.keys():
            path='images/'+key
            print(key)
            images.append(path)
            names.append(res[key])


        return render_template('foundation.html', images=images, names=names)


    return render_template('upload.html', form=form, attr="Foundation")

'''@app.route('/quiz', methods=['GET','POST'])
def quiz():
    form = QuizForm1()
    if request.method == 'POST' and form.validate():
        print(form.hf.data)
        return  "hello world"
    return render_template('test.html', form=form)'''


'''@app.route('/magic_mirror', methods=['GET', 'POST'])
def magic_mirror():
    form = MagicMirror()
    if form.validate_on_submit():
        return redirect(url_for('hello_world'))
    return render_template('magicmirror.html', form=form)'''


@app.route('/change_shade', methods=['GET','POST'])
def change_shade():
    shade_name = request.form['submit_button']
    with open('/Users/aishwarya/PycharmProjects/flaskProject3/dict_lips.pickle', 'rb') as handle:
        shades = pickle.load(handle)

    camera.g=int(shades[shade_name][1])
    camera.r=int(shades[shade_name][0])
    camera.b=int(shades[shade_name][2])
    print(camera.b, camera.g, camera.r )
    return render_template('lipstick.html', images=images, names=names)




@app.route('/')
def hello_world():


    return 'Hello World!'


if __name__ == '__main__':
    app.run()
