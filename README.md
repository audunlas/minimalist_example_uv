# minimalist_example_uv

A CHAP-compatible disease prediction model using linear regression.

## Setup

Install dependencies using uv:

```bash
uv sync
```

## Usage with CHAP

Evaluate the model using CHAP:

```bash
chap evaluate --model-name ./ --dataset-csv your_data.csv
```

## Manual usage

Train the model:

```bash
uv run python main.py train training_data.csv model.pkl
```

Generate predictions:

```bash
uv run python main.py predict model.pkl historic.csv future.csv predictions.csv
```

## Model description

This is a simple linear regression model that predicts disease cases based on:
- rainfall
- mean_temperature

You can modify `main.py` to use different features or a different model architecture.
