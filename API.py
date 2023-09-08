#import re #For Swagger
#import h5py
import numpy as np 
#import keras
#import keras.api._v2.keras as keras
import os
from flask import Flask, app,request,render_template 
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.models import Model

from keras.models import load_model
from keras.utils import load_img, img_to_array
#import keras.utils as image
#from keras.preprocessing import image 
from tensorflow.python.ops.gen_array_ops import concat 
from tensorflow import concat
from keras.applications.inception_v3 import preprocess_input 
import requests 
from flask import Flask, request, jsonify, render_template, redirect, url_for 

from flasgger import Swagger

#Loading the model 
modeln=load_model(r"vgg-16-DOG-CAT-model.h5",compile=False) 
app=Flask(__name__) 
swagger = Swagger(app)
@app.route('/result',methods=["POST"]) 
def nres():
    """
    A simple ML API that returns the animal name & the view of the animal.
    ---
    parameters:
       - name: image
         required: true
         in: formData
         type: file
    requestBody:
     content:
      image/png:
       schema:
        type: string
        format: binary
    responses:
        200:
            description : Detected Animal & its view
    """ 
    if request.method=="POST": 
        f=request.files['image']
        basepath=os.path.dirname(__file__) #getting the current path i.e where app.py is present 
        #print("current path",basepath) 
        filepath=os.path.join(basepath,'uploads',"input") #from anywhere in the system we can give image but we want that image later to process so we are saving it to uploads folder for reusing 
        #print("upload folder is",filepath) 
        f.save(filepath) 
        #img=image.load_img(filepath,target_size=(224,224)) 
        img=load_img(filepath,target_size=(224,224)) 
        #x=image.img_to_array(img)#img to array 
        x=img_to_array(img)#img to array 
        x=np.expand_dims(x,axis=0)#used for adding one more dimension 
        #print(x) 
        img_data=preprocess_input(x) 
        prediction=np.argmax(modeln.predict(img_data)) 
        index=['Frontcat','Frontdog','Leftcat','Leftdog','Rightcat','Rightdog']
        add1 = "This picture captured a "
        add2 = "'s view from the "
        nresult  = add1+str(index[prediction])[-3:]+add2+str(index[prediction])[:-3]+"."
        return jsonify({"Result": nresult})



""" Running our application """ 
if __name__ == "__main__": 
    app.run(debug =False, port = 8080)
