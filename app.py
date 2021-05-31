from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('xgbcl_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        ref_index = float(request.form['ref_index'])
        sodium = float(request.form['sodium'])
        magnesium = float(request.form['magnesium'])
        alum = float(request.form['alum'])
        silicon = float(request.form['silicon'])
        potash = float(request.form['potash'])
        barium = float(request.form['barium'])
        calcium = float(request.form['calcium'])
        iron = float(request.form['iron'])

        params = np.array([[ref_index, sodium, magnesium, alum, silicon, potash, barium, calcium, iron]])
        preds = model.predict(params)
        output = preds
        if output<0:
            return render_template('index.html',prediction_texts="This type of glass does not exist.")
        else:
            if output==1:
                return render_template('index.html',prediction_text="The Type of glass is Building Window (float processed)")
            elif output==2:
                return render_template('index.html',prediction_text="The Type of glass is Building Window (non float processed)")
            elif output==3:
                return render_template('index.html',prediction_text="The Type of glass is Vehicle Window (float processed)")
            elif output==4:
                return render_template('index.html',prediction_text="The Type of glass is Vehicle Window (non float processed)")
            elif output==5:
                return render_template('index.html',prediction_text="The Type of glass is Containers")
            elif output==6:
                return render_template('index.html',prediction_text="The Type of glass is Tableware")
            elif output==7:
                return render_template('index.html',prediction_text="The Type of glass is Headlamps")
            else:
                return render_template('index.html',prediction_text="Unable to classify the type of glass.")
    
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)