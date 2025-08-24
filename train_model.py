import json
import joblib
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the dataset from a CSV file.

    Args:
        csv_path: Path to the input CSV file.

    Returns:
        A preprocessed pandas DataFrame.
        
    Raises:
        ValueError: If required columns are missing from the CSV.
    """
    df = pd.read_csv(csv_path)
    # Drop location if it exists, as it's often redundant with Country
    if "Location" in df.columns:
        df = df.drop(columns=["Location"])
    
    # Clean column names (strip whitespace and replace spaces with underscores)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    
    # Define required columns for the model
    required = ["Country", "Category", "Visitors", "Rating", "Revenue", "Accommodation_Available"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
        
    # Drop rows with missing values in essential columns and ensure correct types
    df = df.dropna(subset=required).copy()
    df["Country"] = df["Country"].astype(str)
    df["Category"] = df["Category"].astype(str)
    df["Accommodation_Available"] = df["Accommodation_Available"].astype(str)
    
    return df

def train_and_save(csv_path: str, out_dir: str = "."):
    """
    Trains a regression model and saves the pipeline, metadata, and feature choices.

    Args:
        csv_path: Path to the training data CSV.
        out_dir: Directory to save the output files ('model.pkl', 'choices.json').
    """
    df = load_data(csv_path)
    
    # Define features (X) and target (y)
    X = df[["Country", "Category", "Visitors", "Rating", "Accommodation_Available"]].copy()
    y = df["Revenue"].astype(float).copy()

    # Create a preprocessing pipeline for numeric and categorical features
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["Visitors", "Rating"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Country", "Category", "Accommodation_Available"]),
        ],
        remainder="passthrough" # Keep other columns if any
    )

    # Define the model pipeline
    model = Ridge(alpha=1.0)
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])

    # Check if the target variable is skewed and apply log transformation if necessary
    use_log = abs(y.skew()) > 1.0
    if use_log:
        reg = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    else:
        reg = pipe

    # Split data, train the model, and evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg.fit(X_train, y_train)
    rmse = float(np.sqrt(mean_squared_error(y_test, reg.predict(X_test))))

    # Create output directory
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the trained model
    joblib.dump(reg, output_dir / "model.pkl")

    # Save feature choices and metadata for the web app
    choices = {
        "Country": sorted(X["Country"].astype(str).unique().tolist()),
        "Category": sorted(X["Category"].astype(str).unique().tolist()),
        "Accommodation_Available": ["Yes", "No"],
        "numeric_ranges": {
            "Visitors": [float(max(0, X["Visitors"].min())), float(X["Visitors"].max())],
            "Rating": [float(max(0, X["Rating"].min())), float(X["Rating"].max())],
        },
        "meta": {
            "model_type": "Ridge",
            "holdout_rmse": round(rmse, 2),
            "used_log_target": bool(use_log),
        },
    }
    with open(output_dir / "choices.json", "w", encoding="utf-8") as f:
        json.dump(choices, f, indent=2)

    print(f"✅ Saved model.pkl and choices.json to '{output_dir}'")
    print(f"📊 Holdout RMSE: {rmse:.2f} | Log target used: {use_log}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tourism revenue prediction model.")
    parser.add_argument("--csv", required=True, help="Path to the tourism dataset CSV file.")
    parser.add_argument("--out", default=".", help="Output directory for model and choices.")
    args = parser.parse_args()
    
    train_and_save(args.csv, args.out)
