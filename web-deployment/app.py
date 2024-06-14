from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("arima_model.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "steps" not in data:
        return jsonify(error="The 'steps' field is required"), 400

    if not isinstance(data["steps"], int):
        return jsonify(error="The 'steps' field must be an integer"), 400

    if data["steps"] < 1:
        return jsonify(error="The 'steps' field must be greater than 0"), 400

    steps = data["steps"]
    forecast = model.forecast(steps=steps)
    forecast_list = forecast.tolist()

    return jsonify(forecast=forecast_list)


if __name__ == "__main__":
    app.run(debug=False)
