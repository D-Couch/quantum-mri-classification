"""Classification metrics with explicit multiclass and undefined-score handling."""


def classification_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str], scores=None
) -> dict:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    if not labels or any(value not in labels for value in y_true + y_pred):
        raise ValueError("all observed labels must be declared in labels")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")
    matrix = [[0 for _ in labels] for _ in labels]
    index = {label: i for i, label in enumerate(labels)}
    for actual, predicted in zip(y_true, y_pred):
        matrix[index[actual]][index[predicted]] += 1

    total = len(y_true)
    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(row[i] for row in matrix) for i in range(len(labels))]
    per_class, f1s, precisions, recalls = {}, [], [], []
    for i, label in enumerate(labels):
        tp, support, predicted_count = matrix[i][i], row_sums[i], col_sums[i]
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        tn = total - support - predicted_count + tp
        specificity_denominator = tn + predicted_count - tp
        specificity = tn / specificity_denominator if specificity_denominator else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    correct = sum(matrix[i][i] for i in range(len(labels)))
    dot = sum(a * b for a, b in zip(row_sums, col_sums))
    denominator = (
        (total**2 - sum(x * x for x in col_sums)) * (total**2 - sum(x * x for x in row_sums))
    ) ** 0.5
    mcc = (correct * total - dot) / denominator if denominator else 0.0
    expected_accuracy = dot / total**2
    observed_accuracy = correct / total
    kappa = (
        (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
        if expected_accuracy < 1
        else 0.0
    )
    result = {
        "accuracy": observed_accuracy,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "weighted_f1": sum(x["f1"] * x["support"] for x in per_class.values()) / total,
        "matthews_correlation_coefficient": mcc,
        "cohen_kappa": kappa,
        "confusion_matrix": matrix,
        "labels": labels,
        "per_class": per_class,
        "roc_auc_ovr_macro": None,
        "pr_auc_macro": None,
    }
    if scores is not None:
        if len(scores) != total or any(len(row) != len(labels) for row in scores):
            raise ValueError("scores must have one value per sample and declared class")
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            y_index = [index[label] for label in y_true]
            result["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_index,
                    scores,
                    labels=list(range(len(labels))),
                    multi_class="ovr",
                    average="macro",
                )
            )
            result["pr_auc_macro"] = float(
                sum(
                    average_precision_score(
                        [int(actual == class_index) for actual in y_index],
                        [score[class_index] for score in scores],
                    )
                    for class_index in range(len(labels))
                )
                / len(labels)
            )
        except ValueError:
            # AUC is undefined if a partition lacks positives or score rows are invalid.
            pass
    return result


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    return classification_metrics(y_true, y_pred, labels)["macro_f1"]
