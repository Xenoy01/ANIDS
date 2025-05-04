from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()
        input_df = pd.DataFrame([input_data])

        # Ensure correct format
        input_df = pd.get_dummies(input_df)

        # Align with training data columns
        model_features = joblib.load('model_columns.pkl')  # Save model columns during training
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        prediction = model.predict(input_df)
        return render_template('index.html', prediction_text=f'Prediction: {prediction[0]}')

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
