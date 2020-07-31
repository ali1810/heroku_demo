import cv2
import matplotlib.image as mpimg
from keras.models import load_model
import numpy as np
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/CAE_100_lungs_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))

    # Preprocessing the image
    img1=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
     # img1 =cv2.resize(img1,(64,64)) 
     #print(img1)
    img1 = img1.astype('float32') / 255.
    #print(img1)
    #image2 = np.reshape(image1,[1,64,64,3])
    img1 = np.reshape(img1, [1,64,64,3])
    preds = model.predict(img1)
    return preds
    
    
    
    
    #x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    #preds = model.predict(x)
    #return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
         f = request.files['file']

        # Save the file to ./uploads
         basepath = os.path.dirname(__file__)
         file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
         f.save(file_path)

        # Make prediction
         preds = model_predict(file_path, model)
	
	 mse = np.mean((img1 - preds) ** 2)
         label = "Anomalous" if mse < 0.1045 else "normal"
         color = (0, 0, 255) if mse > 0.1045 else (0, 255, 0)

        # draw the predicted label text on the original image
         cv2.putText(image, label, (10,  25), cv2.FONT_HERSHEY_SIMPLEX,
	  0.7,color,2)
         # display the image
         plt.imshow("Output", image)
         plt.close
         cv2.waitKey(0)
   

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        #return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
