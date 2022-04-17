from flask import Flask, request, render_template
import pickle
import requests

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb')) 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    rooms = int(request.form["rooms"])
    distance = int(request.form["distance"])
    prediction = model.predict([[rooms, distance]])  
    output = round(prediction[0], 2) 

    return render_template('index.html', prediction_text=f'A house with {rooms} rooms and located {distance} meters from the city center has a value of ${output}')

    
if __name__ == "__main__":
    app.run()

