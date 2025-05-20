import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify
import os

model_path = os.path.join(os.path.dirname(__file__), '../models/best_random_forest_model.joblib')
model = joblib.load(model_path)

app = Flask(__name__)
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        glucose = data.get("glucose")
        hba1c = data.get("hba1c")

        print(f"Features received: glucose={glucose}, hba1c={hba1c}")

        if glucose is None or hba1c is None:
            return jsonify({"error": "Both glucose and hba1c values are required."})

        try:
            glucose = float(glucose)
            hba1c = float(hba1c)
        except ValueError:
            return jsonify({"error": "Invalid input. Please enter numeric values."})

        input_data = np.array([[glucose, hba1c]])
        
        proba = model.predict_proba(input_data)[:, 1]
        result = ["Positivo" if p >= 0.7 else "Negativo" for p in proba]

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)