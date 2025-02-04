import numpy as np
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compare_metrics(df, len_forecast, target_column):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_SIZES = ["tiny", "mini", "small", "base", "large"]

    len_context = len(df) - len_forecast
    context = torch.tensor(df[target_column])[:len_context]
    actual_values = df[target_column][len_context : len_context + len_forecast].values  # Ground truth values

    for size in MODEL_SIZES:
        model_name = f"amazon/chronos-t5-{size}"

        # Load pipeline with the current model
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=DEVICE,
            torch_dtype=torch.bfloat16,
        )

        # Get predictions
        forecast = pipeline.predict(context, len_forecast)

        # Compute prediction quantiles
        forecast_index = range(len_context, len_context + len_forecast)
        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

        # Compute error metrics
        mae = mean_absolute_error(actual_values, median) # Mean absolute error
        mse = mean_squared_error(actual_values, median) # Mean squared error
        rmse = np.sqrt(mse) # Root mean squared error
        mape = np.mean(np.abs((actual_values - median) / actual_values)) * 100  # Mean absolute percentage error
        wql = np.mean(np.abs((actual_values - median) / (high - low))) * 100  # Weighted quantile loss

        # Print metrics
        print(f"Evaluated Chronos {size} - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, WQL: {wql:.2f}%")