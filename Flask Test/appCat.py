from flask import Flask, request, render_template
import pickle
import requests

app = Flask(__name__)
model = pickle.load(open('modelCats.pkl', 'rb')) 

@app.route('/')
def home():
    return render_template('indexCat.html')

@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    kg = int(request.form["kg"])
    type = int(request.form["type"])
    prediction = model.predict([[kg, type]])  
    output = round(prediction[0], 2) 

    return render_template('indexCat.html', prediction_text=f'A fat Perla(cat) that weights {kg} kg and is type {type} must eat around {output} calories per day')

    
if __name__ == "__main__":
    app.run()

