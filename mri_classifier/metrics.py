"""Dependency-light classification metrics for small experiments and tests."""


def classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    if not labels or any(v not in labels for v in y_true + y_pred):
        raise ValueError("all observed labels must be declared in labels")
    matrix = [[0 for _ in labels] for _ in labels]
    index = {label: i for i, label in enumerate(labels)}
    for actual, predicted in zip(y_true, y_pred):
        matrix[index[actual]][index[predicted]] += 1
    per_class = {}
    f1s = []
    total = len(y_true)
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1s.append(f1)
    correct = sum(matrix[i][i] for i in range(len(labels)))
    return {
        "accuracy": correct / total,
        "macro_f1": sum(f1s) / len(f1s),
        "weighted_f1": sum(x["f1"] * x["support"] for x in per_class.values()) / total,
        "confusion_matrix": matrix,
        "labels": labels,
        "per_class": per_class,
    }


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    return classification_metrics(y_true, y_pred, labels)["macro_f1"]
