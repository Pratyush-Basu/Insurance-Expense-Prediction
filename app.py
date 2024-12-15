from flask import Flask, render_template, request
import joblib
import numpy as np
import sklearn
print(sklearn.__version__)

app = Flask(__name__)

# Load the trained machine learning model with double backslashes
model = joblib.load("D:\\Programming\\ML porjects\\insurence\\model\\model.pkl")

@app.route("/")
def home():
    # Render the home page template
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the form
    age = float(request.form["age"])
    sex = request.form["sex"]
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])
    smoker = request.form["smoker"]
    region = request.form["region"]
    
    # Convert categorical variables into numerical values
    sex = 1 if sex == "male" else 0
    smoker = 1 if smoker == "yes" else 0
    
    # Encode region
    if region == "southwest":
        region_encoded = 1
    elif region == "southeast":
        region_encoded = 2
    elif region == "northwest":
        region_encoded = 3
    elif region == "northeast":
        region_encoded = 4
    else:
        region_encoded = 0  # Default value if region doesn't match any of the specified regions

    # Combine input features into a single array
    features = np.array([[age, sex, bmi, children, smoker, region_encoded]])

    # Make prediction using the loaded model
    prediction = model.predict(features)

    # Return the prediction result to the user
    return render_template("result.html", prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
