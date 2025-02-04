import matplotlib.pyplot as plt
import numpy as np
import torch
from chronos import ChronosPipeline

def compare_plots(df, len_forecast, target_column):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_SIZES = ["tiny", "mini", "small", "base", "large"]

    len_context = len(df) - len_forecast
    context = torch.tensor(df[target_column])[:len_context]
    actual_values = df[target_column][len_context : len_context + len_forecast].values  # Ground truth values

    # Define colors and markers
    colors = ["royalblue", "green", "red", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "h", "D"]

    plt.figure(figsize=(8, 4))
    plt.plot(range(len_context, len_context + len_forecast), actual_values, color="royalblue", label="historical data")

    for i, size in enumerate(MODEL_SIZES):
        print(f"Evaluating Chronos {size} model...")
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

        # Plot the results with different colors and markers
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.plot(forecast_index, median, label=f"{size} median forecast", color=color, marker=marker)
        plt.fill_between(forecast_index, low, high, alpha=0.3, color=color, label=f"{size} 80% prediction interval")

    # Add title, legend, and grid
    plt.title("Forecast using different Chronos model sizes")
    plt.legend()
    plt.grid()
    plt.show()