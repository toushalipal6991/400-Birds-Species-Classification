# import tensorflow
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_prep
import cv2
from flask import Flask,redirect, url_for, request, render_template


import flask
app = Flask(__name__)

model = load_model("weights-55-0.2253.h5")
class_indices = pickle.load( open("class_indices.pkl","rb") )
idx_species = {v:k for k,v in class_indices.items()}


###################################################
#Adding image pre-processing function
def predict_bird_species(img_path):

    # 55x55 input
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = vgg19_prep(img_to_array(img))
    img = np.expand_dims(img, axis=0)


    # 224x224 input
    # # load the image
    # img = load_img(img_path, target_size=(224, 224))  #(224,224,3)
    # # convert to array
    # img = img_to_array(img) #(224,224,3)
    # # preprocess
    # img = vgg19_prep(img)
    # # add batch size as a dimension 
    # img = np.expand_dims(img, axis=0)  #(1,224,224,3)

    prediction_probability = model.predict(img)
    top = np.argsort(prediction_probability)[0][::-1][0]
    predicted_species = idx_species.get(top)

    return "This bird belongs to " + predicted_species + " species"

###################################################

@app.route('/', methods=['GET'])
def welcome():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        imagefile=request.files['imagefile']
        if imagefile:
            image_path = "./static/" + imagefile.filename
            imagefile.save(image_path)
            return render_template('index.html',prediction=predict_bird_species(image_path),imageloc=imagefile.filename)
    return render_template('index.html',prediction=predict_bird_species(image_path),imageloc = None)


if __name__ == "__main__":
	app.run(port=8080)


#References:-
#https://www.youtube.com/watch?v=0nr6TPKlrN0&ab_channel=Jay    