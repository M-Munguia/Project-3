import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

flask_app = Flask(__name__)
XGB_model = pickle.load(open('XGBmodel.pkl', "rb")) # load ml model



# The route() decorator to tell Flask what URL should trigger our function.
# ‘/’ is the root of the website, such as www.westga.edu
@flask_app.route("/")   
def index():
    return render_template("index.html")


@flask_app.route("/predict", methods = ["POST"])   
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    result = XGB_model.predict(features)
    return render_template("index.html", predicted_text = result)

if __name__ =="__main__":
    flask_app.run(debug = True)
