"""CHAP model: minimalist_example_uv

A simple linear regression model for disease prediction.
"""

import joblib
import pandas as pd
from cyclopts import App
from sklearn.linear_model import LinearRegression

app = App()


@app.command()
def train(train_data: str, model: str):
    """Train the model on the provided data.

    Parameters
    ----------
    train_data
        Path to the training data CSV file.
    model
        Path where the trained model will be saved.
    """
    df = pd.read_csv(train_data)
    features = df[["rainfall", "mean_temperature"]].fillna(0)
    target = df["disease_cases"].fillna(0)

    reg = LinearRegression()
    reg.fit(features, target)
    joblib.dump(reg, model)
    print(f"Model saved to {model}")


@app.command()
def predict(model: str, historic_data: str, future_data: str, out_file: str):
    """Generate predictions using the trained model.

    Parameters
    ----------
    model
        Path to the trained model file.
    historic_data
        Path to historic data CSV file (unused in this simple model).
    future_data
        Path to future climate data CSV file.
    out_file
        Path where predictions will be saved.
    """
    reg = joblib.load(model)
    future_df = pd.read_csv(future_data)
    features = future_df[["rainfall", "mean_temperature"]].fillna(0)

    predictions = reg.predict(features)
    output_df = future_df[["time_period", "location"]].copy()
    output_df["sample_0"] = predictions
    output_df.to_csv(out_file, index=False)
    print(f"Predictions saved to {out_file}")


if __name__ == "__main__":
    app()
