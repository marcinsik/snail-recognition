import os
from PIL import Image
import numpy as np
from keras import models
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'static/upload'

model = models.load_model('model/m1_8_20.h5')
classes = ['Blotniarka', 'Helenka', 'Winniczek', 'Negative']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        filename = 'upload_img.jpg'
        return render_template('home.html', filename=filename)
    if request.method == "POST":
        image_upload = request.files['image_upload']
        if image_upload.filename != '':
            image_name = secure_filename(image_upload.filename)
            file_path = os.path.join(app.config['UPLOAD_PATH'], image_name)
            image = Image.open(image_upload)
            image = image.resize((300, 300))
            image.save(file_path)
            image_array = np.array(image.convert('RGB'))
            image_array.shape = (1, 300, 300, 3)
            result = model.predict(image_array)
            res = np.argmax(result)
            return render_template('home.html', prediction=classes[res], filename=image_name)

if __name__ == '__main__':
    app.run()