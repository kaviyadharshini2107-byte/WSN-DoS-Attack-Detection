from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test, class_names):

    predictions = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_precision": precision_score(
            y_test, predictions, average="macro"
        ),
        "macro_recall": recall_score(
            y_test, predictions, average="macro"
        ),
        "macro_f1": f1_score(
            y_test, predictions, average="macro"
        )
    }

    print("MODEL EVALUATION")
    print("----------------")

    for metric, value in results.items():
        print(f"{metric}: {value * 100:.4f}%")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        target_names=class_names
    ))

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_test, predictions))

    return results