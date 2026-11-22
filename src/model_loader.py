
import joblib
import os


def load_final_model(model_path):
    """
    Load the saved GOA-GA optimized CART model.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    saved_data = joblib.load(model_path)

    model = saved_data["model"]
    selected_features = saved_data["selected_features"]

    print("FINAL MODEL LOADED")
    print("-------------------")
    print("Model:", type(model).__name__)
    print("Selected features:", selected_features)
    print("Number of features:", len(selected_features))
    print(
        "Test accuracy:",
        round(saved_data["test_accuracy"] * 100, 4),
        "%"
    )

    return model, selected_features
