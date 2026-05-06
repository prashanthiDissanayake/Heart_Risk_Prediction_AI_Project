from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from utils.preprocessing import preprocess_input

app = Flask(__name__)
CORS(app)

model = joblib.load("models/xgb_model.pkl")


def get_risk_level(probability):
    if probability < 0.25:
        return "Low Risk"
    elif probability < 0.50:
        return "Medium Risk"
    else:
        return "High Risk"


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Heart Disease Prediction API is running successfully"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_data = preprocess_input(data)

        prediction = model.predict(input_data)[0]
        disease_probability = model.predict_proba(input_data)[0][1]

        predicted_label = "Disease" if prediction == 1 else "No Disease"
        risk_level = get_risk_level(disease_probability)

        return jsonify({
            "predicted_class": int(prediction),
            "predicted_label": predicted_label,
            "disease_probability": round(float(disease_probability) * 100, 2),
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)