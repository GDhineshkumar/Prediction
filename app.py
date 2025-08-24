import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

# --- Initialization ---
app = Flask(__name__)

# --- Load Model and Choices ---
try:
    # Load the trained machine learning model
    model = joblib.load("model.pkl")
    print("✅ Model loaded successfully.")
    
    # Load the choices/metadata file for populating the form
    with open("choices.json", "r", encoding="utf-8") as f:
        choices = json.load(f)
    print("✅ Choices loaded successfully.")

except FileNotFoundError as e:
    print(f"🚨 Error loading files: {e}")
    print("Please run train.py first to generate model.pkl and choices.json.")
    model = None
    choices = None
except Exception as e:
    print(f"🚨 An unexpected error occurred: {e}")
    model = None
    choices = None

# --- Helper Function ---
def make_dataframe(form_data: dict) -> pd.DataFrame:
    """
    Converts form data from the web request into a pandas DataFrame
    that matches the model's expected input format.
    
    Args:
        form_data: A dictionary containing the user's input from the form.
        
    Returns:
        A pandas DataFrame with a single row for prediction.
    """
    # Create a dictionary with the correct structure and data types
    data_dict = {
        "Country": [form_data.get("country")],
        "Category": [form_data.get("category")],
        "Visitors": [float(form_data.get("visitors"))],
        "Rating": [float(form_data.get("rating"))],
        "Accommodation_Available": [form_data.get("accommodation")],
    }
    # Convert the dictionary to a pandas DataFrame
    return pd.DataFrame(data_dict)

# --- App Routes ---
@app.route("/", methods=["GET"])
def index():
    """Renders the main page with the prediction form."""
    if not choices:
        return "Model and choices not loaded. Please check the server logs.", 500
        
    return render_template(
        "index.html",
        countries=choices.get("Country", []),
        categories=choices.get("Category", []),
        acc_options=choices.get("Accommodation_Available", []),
        ranges=choices.get("numeric_ranges", {}),
        meta=choices.get("meta", {})
    )

@app.route("/predict", methods=["POST"])
def predict():
    """Handles the form submission, makes a prediction, and shows the result."""
    if not model:
        return render_template("result.html", prediction=None, error="Model is not loaded.")
        
    try:
        # Create a DataFrame from the submitted form data
        X = make_dataframe(request.form)
        
        # Use the loaded model to make a prediction
        prediction = model.predict(X)[0]
        
        # Format the prediction for display
        formatted_prediction = f"${prediction:,.2f}"
        
        return render_template("result.html", prediction=formatted_prediction, error=None)
    except Exception as e:
        # Handle potential errors during prediction (e.g., invalid input)
        error_message = f"An error occurred: {e}"
        return render_template("result.html", prediction=None, error=error_message)

if __name__ == "__main__":
    # Get port from environment variable or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Run the app
    app.run(host="0.0.0.0", port=port, debug=True)
