from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route("/", methods=['GET'])

def Home():    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
         age = int(request.form['age'])
         JobType=request.form['JobType']
         EdType=request.form['EdType']
         maritalstatus=request.form['maritalstatus']
         occupation=request.form['occupation']
         relationship=request.form['relationship']
         race=request.form['race']
         gender=request.form['gender']
         capitalgain=int(request.form['capitalgain'])
         capitalloss=int(request.form['capitalloss'])
         hoursperweek=int(request.form['hoursperweek'])
         
         SampleInputData=pd.DataFrame(
            data=[[age,JobType,EdType,maritalstatus,occupation,relationship,race,gender,capitalgain,capitalloss,hoursperweek]],
            columns=['age', 'JobType', 'EdType', 'maritalstatus', 'occupation','relationship', 'race', 'gender', 'capitalgain', 'capitalloss',
                     'hoursperweek'])
         Num_Inputs=SampleInputData.shape[0]
         DataForML= pickle.load(open('DataForML.pkl', 'rb'))
         SampleInputData=SampleInputData.append(DataForML)
         SampleInputData['gender'].replace({' Female':0, ' Male':1}, inplace=True)
         SampleInputData=pd.get_dummies(SampleInputData)
         X=SampleInputData.values[0:Num_Inputs]

         with open('FinalLogisticModel.pkl', 'rb') as fileReadStream:
                Final_model=pickle.load(fileReadStream)
                fileReadStream.close()
                Prediction=Final_model.predict(X)

         if Prediction == 0:
             return render_template('Result.html',prediction_texts="Prediction: Income is Greater than 50K")
         else:
            return render_template('Result.html',prediction_texts="{Prediction: Income is less than or equal to 50K")  
       
    else:   
        return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)