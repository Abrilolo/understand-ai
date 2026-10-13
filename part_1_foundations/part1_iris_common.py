"""
Shared code for the Iris experiments in Part I.

The neural network math in this file is pure Python:
- no TensorFlow
- no PyTorch
- no scikit-learn model

We only use scikit-learn to load the public Iris dataset, because the task
recommends Iris and the project already uses scikit-learn for dataset loading.
"""

import math
from pathlib import Path


CLASS_NAMES = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = [
    "sepal length",
    "sepal width",
    "petal length",
    "petal width",
]
CLASS_COLORS = ["#2f80ed", "#27ae60", "#eb5757"]


def sigmoid(value):
    """Squash any real number into a smooth 0..1 value."""
    return 1 / (1 + math.exp(-value))


def sigmoid_derivative(output):
    """Derivative of sigmoid, written in terms of the already-known output."""
    return output * (1 - output)


def softmax(values):
    """
    Convert raw output scores into probabilities that sum to 1.

    The largest score becomes the largest probability, which lets us pick one
    of the three Iris classes by taking the index with the highest probability.
    """
    max_value = max(values)
    exponentials = [math.exp(value - max_value) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def one_hot(class_index, class_count=3):
    """Represent one class as a target vector such as [0, 1, 0]."""
    values = [0] * class_count
    values[class_index] = 1
    return values


def argmax(values):
    """Return the index of the largest value without using NumPy."""
    best_index = 0
    best_value = values[0]

    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value

    return best_index


def load_iris_rows():
    """Load Iris rows as plain Python lists."""
    try:
        from sklearn.datasets import load_iris
    except ImportError as exc:
        raise SystemExit(
            "This script uses scikit-learn only to load the Iris dataset.\n"
            "Install it with:\n"
            "  .venv\\Scripts\\python.exe -m pip install scikit-learn"
        ) from exc

    iris = load_iris()
    rows = []

    for features, target in zip(iris.data, iris.target):
        rows.append(
            {
                "features": [float(value) for value in features],
                "target": int(target),
            }
        )

    return rows


def split_train_test(rows):
    """
    Create a deterministic stratified split.

    Iris has 50 examples per class. We put every fifth example from each class
    in the test set, so each class contributes 10 test rows and 40 train rows.
    """
    train_rows = []
    test_rows = []
    seen_by_class = {class_index: 0 for class_index in range(3)}

    for row in rows:
        class_index = row["target"]
        class_position = seen_by_class[class_index]
        seen_by_class[class_index] += 1

        if class_position % 5 == 0:
            test_rows.append(row)
        else:
            train_rows.append(row)

    return train_rows, test_rows


def feature_stats(rows):
    """Compute mean and standard deviation from the training split only."""
    feature_count = len(rows[0]["features"])
    means = []
    standard_deviations = []

    for feature_index in range(feature_count):
        values = [row["features"][feature_index] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        standard_deviation = math.sqrt(variance) or 1
        means.append(mean)
        standard_deviations.append(standard_deviation)

    return means, standard_deviations


def normalize_features(features, means, standard_deviations):
    """Scale features so training is smoother and no feature dominates unfairly."""
    return [
        (value - means[index]) / standard_deviations[index]
        for index, value in enumerate(features)
    ]


def normalize_rows(rows, means, standard_deviations):
    """Return rows with normalized features while keeping the original features."""
    normalized_rows = []

    for row in rows:
        normalized_rows.append(
            {
                "features": normalize_features(
                    row["features"], means, standard_deviations
                ),
                "original_features": row["features"],
                "target": row["target"],
            }
        )

    return normalized_rows


class TinyIrisNetwork:
    """
    A tiny 4-H-3 neural network.

    4 input values:
        sepal length, sepal width, petal length, petal width

    H hidden neurons:
        controlled by the script that imports this shared file

    3 output neurons:
        one score for setosa, one for versicolor, one for virginica
    """

    def __init__(self, hidden_neurons):
        self.hidden_neurons = hidden_neurons
        self.input_count = 4
        self.output_count = 3

        # Deterministic, non-random initial weights.
        # The formula gives each neuron different small values so the hidden
        # neurons do not all learn the exact same thing.
        self.hidden_weights = [
            [
                0.15 * math.sin((hidden_index + 1) * (input_index + 2))
                for input_index in range(self.input_count)
            ]
            for hidden_index in range(hidden_neurons)
        ]
        self.hidden_biases = [
            0.05 * math.cos(hidden_index + 1)
            for hidden_index in range(hidden_neurons)
        ]
        self.output_weights = [
            [
                0.15 * math.sin((output_index + 2) * (hidden_index + 1))
                for hidden_index in range(hidden_neurons)
            ]
            for output_index in range(self.output_count)
        ]
        self.output_biases = [
            0.05 * math.cos(output_index + 2)
            for output_index in range(self.output_count)
        ]

    def forward(self, inputs):
        """Run one prediction and return hidden activations plus probabilities."""
        hidden_outputs = []

        for weights, bias in zip(self.hidden_weights, self.hidden_biases):
            weighted_sum = sum(
                input_value * weight
                for input_value, weight in zip(inputs, weights)
            )
            hidden_outputs.append(sigmoid(weighted_sum + bias))

        raw_outputs = []
        for weights, bias in zip(self.output_weights, self.output_biases):
            weighted_sum = sum(
                hidden_output * weight
                for hidden_output, weight in zip(hidden_outputs, weights)
            )
            raw_outputs.append(weighted_sum + bias)

        probabilities = softmax(raw_outputs)
        return hidden_outputs, probabilities

    def predict(self, inputs):
        """Predict the Iris class with the highest output probability."""
        _, probabilities = self.forward(inputs)
        return argmax(probabilities)

    def train_step(self, inputs, target_class, learning_rate):
        """
        Train on one row with backpropagation.

        The output layer uses softmax + cross entropy. With that combination,
        the output delta is beautifully simple:

            delta = predicted_probability - target_probability

        Then the output deltas are pushed backward into the hidden layer.
        """
        hidden_outputs, probabilities = self.forward(inputs)
        target = one_hot(target_class, self.output_count)

        # Cross entropy loss for the correct class.
        loss = -math.log(probabilities[target_class] + 1e-12)

        output_deltas = [
            probabilities[index] - target[index]
            for index in range(self.output_count)
        ]

        old_output_weights = [weights[:] for weights in self.output_weights]

        # Update output layer weights and biases.
        for output_index in range(self.output_count):
            for hidden_index, hidden_output in enumerate(hidden_outputs):
                gradient = output_deltas[output_index] * hidden_output
                self.output_weights[output_index][hidden_index] -= (
                    learning_rate * gradient
                )
            self.output_biases[output_index] -= learning_rate * output_deltas[output_index]

        # Update hidden layer weights and biases.
        for hidden_index, hidden_output in enumerate(hidden_outputs):
            hidden_error = sum(
                output_deltas[output_index] * old_output_weights[output_index][hidden_index]
                for output_index in range(self.output_count)
            )
            hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

            for input_index, input_value in enumerate(inputs):
                gradient = hidden_delta * input_value
                self.hidden_weights[hidden_index][input_index] -= (
                    learning_rate * gradient
                )
            self.hidden_biases[hidden_index] -= learning_rate * hidden_delta

        return loss


def train_network(network, train_rows, epochs, learning_rate):
    """Train the network for a fixed number of epochs."""
    history = []

    for _ in range(epochs):
        total_loss = 0

        for row in train_rows:
            total_loss += network.train_step(
                row["features"], row["target"], learning_rate
            )

        history.append(total_loss / len(train_rows))

    return history


def evaluate(network, rows):
    """Return accuracy, confusion matrix, and per-row predictions."""
    correct = 0
    confusion = [[0 for _ in range(3)] for _ in range(3)]
    predictions = []

    for row in rows:
        prediction = network.predict(row["features"])
        target = row["target"]
        confusion[target][prediction] += 1
        correct += int(prediction == target)
        predictions.append({**row, "prediction": prediction})

    accuracy = correct / len(rows)
    return accuracy, confusion, predictions


def print_confusion_matrix(confusion):
    """Print rows as real classes and columns as predicted classes."""
    print("Confusion matrix: rows=true class, columns=predicted class")
    print(" " * 14 + " ".join(f"{name[:4]:>6}" for name in CLASS_NAMES))

    for class_index, row in enumerate(confusion):
        values = " ".join(f"{value:>6}" for value in row)
        print(f"  {CLASS_NAMES[class_index]:<12}{values}")


def load_matplotlib():
    """Load matplotlib only for plotting the results."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "This script needs matplotlib only for plots. Install it with:\n"
            "  .venv\\Scripts\\python.exe -m pip install matplotlib"
        ) from exc

    return plt


def plot_predictions(predictions, title, output_path):
    """
    Plot true labels and model predictions using petal length vs petal width.

    The network uses all 4 Iris features. The plot uses 2 features because a
    screen is flat, and petal length/width are the easiest Iris features to see.
    """
    plt = load_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    panels = [
        ("True test labels", "target"),
        ("Network predictions", "prediction"),
    ]

    for axis, (panel_title, key) in zip(axes, panels):
        for class_index, class_name in enumerate(CLASS_NAMES):
            matching_rows = [
                row for row in predictions if row[key] == class_index
            ]
            axis.scatter(
                [row["original_features"][2] for row in matching_rows],
                [row["original_features"][3] for row in matching_rows],
                s=85,
                color=CLASS_COLORS[class_index],
                edgecolor="#111111",
                label=class_name,
            )

        wrong_rows = [
            row for row in predictions if row["target"] != row["prediction"]
        ]
        axis.scatter(
            [row["original_features"][2] for row in wrong_rows],
            [row["original_features"][3] for row in wrong_rows],
            s=150,
            facecolors="none",
            edgecolors="#111111",
            linewidths=2.2,
            marker="s",
            label="wrong" if wrong_rows else None,
        )

        axis.set_title(panel_title)
        axis.set_xlabel("petal length")
        axis.grid(True, alpha=0.25)

    axes[0].set_ylabel("petal width")
    axes[1].legend(loc="lower right")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_iris_experiment(hidden_neurons, epochs, learning_rate, plot_filename):
    """Run one Iris experiment and save its plot."""
    rows = load_iris_rows()
    train_rows, test_rows = split_train_test(rows)
    means, standard_deviations = feature_stats(train_rows)
    train_rows = normalize_rows(train_rows, means, standard_deviations)
    test_rows = normalize_rows(test_rows, means, standard_deviations)

    network = TinyIrisNetwork(hidden_neurons=hidden_neurons)
    history = train_network(network, train_rows, epochs, learning_rate)

    train_accuracy, _, _ = evaluate(network, train_rows)
    test_accuracy, confusion, test_predictions = evaluate(network, test_rows)

    print(f"Iris network: 4 -> {hidden_neurons} -> 3")
    print("=" * 28)
    print(f"train rows: {len(train_rows)}")
    print(f"test rows:  {len(test_rows)}")
    print(f"epochs:     {epochs}")
    print(f"lr:         {learning_rate}")
    print(f"loss:       start={history[0]:.6f}, end={history[-1]:.6f}")
    print(f"train acc:  {train_accuracy * 100:.2f}%")
    print(f"test acc:   {test_accuracy * 100:.2f}%")
    print()
    print_confusion_matrix(confusion)
    print()
    print("Misclassified test rows:")
    wrong_predictions = [
        row for row in test_predictions if row["target"] != row["prediction"]
    ]

    if not wrong_predictions:
        print("  none")
    else:
        for row in wrong_predictions:
            petal_length = row["original_features"][2]
            petal_width = row["original_features"][3]
            print(
                f"  true={CLASS_NAMES[row['target']]:<10} "
                f"pred={CLASS_NAMES[row['prediction']]:<10} "
                f"petal_length={petal_length:.1f}, petal_width={petal_width:.1f}"
            )

    print()
    print("Sample test predictions:")

    for row in test_predictions[:10]:
        probabilities = network.forward(row["features"])[1]
        probability_text = ", ".join(
            f"{CLASS_NAMES[index]}={value:.2f}"
            for index, value in enumerate(probabilities)
        )
        print(
            f"  true={CLASS_NAMES[row['target']]:<10} "
            f"pred={CLASS_NAMES[row['prediction']]:<10} "
            f"({probability_text})"
        )

    plot_path = Path(__file__).parent / "assets" / "plots" / plot_filename
    title = f"Iris Test Set Predictions, 4 -> {hidden_neurons} -> 3"
    plot_predictions(test_predictions, title, plot_path)
    print()
    print(f"Plot saved to: {plot_path}")
