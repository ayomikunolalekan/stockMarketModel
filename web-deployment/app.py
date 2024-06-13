from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved ARIMA model
model = joblib.load("arima_model.pkl")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    data = request.get_data()

    if "steps" not in data:
        return jsonify(error="The 'steps' field is required"), 400

    steps = data["steps"]
    forecast = model.predict(n_periods=steps)
    forecast_list = forecast.tolist()

    return jsonify(forecast=forecast_list)


if __name__ == "__main__":
    app.run(debug=False)
