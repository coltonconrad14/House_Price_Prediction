import argparse
from dataclasses import replace

from house_price_prediction.config import load_settings
from house_price_prediction.model import train_and_save_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model from the configured dataset and save artifact.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=25,
        help="Minimum dataset rows required before training starts.",
    )
    args = parser.parse_args()

    settings = load_settings()
    if args.min_rows > 0:
        settings = replace(settings, training_min_rows=args.min_rows)

    metrics = train_and_save_model(settings)

    print("Training complete")
    print(f"Saved model: {settings.model_path}")
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
