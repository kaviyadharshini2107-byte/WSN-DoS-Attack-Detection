
import pandas as pd


def predict_attack(
    model,
    input_data,
    feature_names,
    selected_features,
    scaler,
    label_encoder
):
    """
    Predict the attack category for a new WSN record.
    """

    # Arrange all input features in training order
    input_df = pd.DataFrame(
        [input_data],
        columns=feature_names
    )

    # Apply the same scaling used during model training
    scaled_data = scaler.transform(input_df)

    # Select the six optimized features
    selected_indices = [
        feature_names.index(feature)
        for feature in selected_features
    ]

    selected_data = scaled_data[:, selected_indices]

    # Predict the attack category
    prediction = model.predict(selected_data)

    # Convert encoded label into original class name
    predicted_class = label_encoder.inverse_transform(prediction)

    return predicted_class[0]
