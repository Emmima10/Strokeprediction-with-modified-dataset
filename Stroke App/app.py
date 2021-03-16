import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler as scaler
from tensorflow import keras



app = Flask(__name__)
#scaler = pickle.load(open('scaler.pkl', 'rb'))
#model =(open('new_model.h5','rb'))
model = keras.models.load_model("final.h5")
scaler = joblib.load('sc_scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    features=[]
    for x in request.form.values():
	    features.append(x)
    if features[0]=="male":
        features[0]=1
    elif features[0]=="female":
        features[0]=0
    else:
        features[0]=2

    if features[4]=="yes":
        features[4]=1
    else:
        features[4]=0
    if features[5]=="urban":
        features[5]=1
    else:
        features[5]=0

    if features[8]=="formely smoked":
        features[8]=1
    elif features[8]=="never smokes":
        features[8]=2
    else:
        features[8]=3

    final_features = np.array(features)
    final_features =final_features.reshape(1,-1)
    final_features = scaler.transform(final_features)    
    prediction = model.predict(final_features)
    output = np.argmax(prediction[0])
    #if (output)==1:
        #predicition_text="ABNORMAL"
    #else:
        #prediction_text="NORMAL"

    if output == 1:
        return render_template('index.html',prediction_text='THE PATIENT IS LIKELY TO HAVE A STROKE')
    
    else:
        return render_template('index.html', prediction_text='THE PATIENT WILL NOT HAVE A STROKE')
        
#@app.route('/predict_api',methods=['POST'])
#def results():

 #   data = request.get_json(force=True)
  #  prediction = model.predict([np.array(list(data.values()))])

   # output = prediction[0]
    #return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)
