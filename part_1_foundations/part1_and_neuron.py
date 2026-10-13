"""
Part I - The Ground: a single neuron that learns the AND gate.

This file intentionally solves only the small AND-gate task from Part I.
It does not implement XOR, a multi-layer network, or the later dataset tasks.
"""

import math
import random
from pathlib import Path


AND_DATASET = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]


def sigmoid(value):
    """Squash any real number into the range 0..1."""
    return 1 / (1 + math.exp(-value))


def sigmoid_derivative(output):
    """
    Compute the derivative of sigmoid using the already-computed output.

    If output = sigmoid(z), then sigmoid'(z) = output * (1 - output).
    This is useful during backpropagation because the neuron already knows
    what it produced during the forward pass.
    """
    return output * (1 - output)


class Neuron:
    """
    A single sigmoid neuron.

    The neuron receives two inputs: x1 and x2.
    It computes:

        z = (w1 * x1) + (w2 * x2) + bias
        output = sigmoid(z)

    For the AND gate, we want output near 1 only when both x1 and x2 are 1.
    """

    def __init__(self, input_count):
        self.weights = [random.uniform(-1, 1) for _ in range(input_count)]
        self.bias = random.uniform(-1, 1)
        self.last_inputs = None
        self.last_output = None

    def forward(self, inputs):
        """Run one forward pass and remember values needed for training."""
        z = sum(weight * value for weight, value in zip(self.weights, inputs))
        z += self.bias

        self.last_inputs = inputs
        self.last_output = sigmoid(z)
        return self.last_output

    def train_step(self, inputs, target, learning_rate):
        """
        Run one backpropagation update for one training example.

        We use squared error:

            loss = 0.5 * (target - output)^2

        Backpropagation asks: "How much did each weight contribute to the
        error?" For one neuron, the chain rule is small enough to write by hand:

            d_loss/d_weight = (output - target) * sigmoid_derivative(output) * input

        Then gradient descent moves each weight in the opposite direction:

            weight = weight - learning_rate * d_loss/d_weight
        """
        output = self.forward(inputs)
        error = output - target
        delta = error * sigmoid_derivative(output)

        for index, input_value in enumerate(inputs):
            gradient = delta * input_value
            self.weights[index] -= learning_rate * gradient

        self.bias -= learning_rate * delta

        return 0.5 * (target - output) ** 2

    def predict(self, inputs):
        """Convert the sigmoid output into a hard class using 0.5 as threshold."""
        output = self.forward(inputs)
        return 1 if output >= 0.5 else 0


def train_and_neuron(epochs=10000, learning_rate=.070):
    """Train one neuron until it behaves like an AND gate."""
    random.seed(7)
    neuron = Neuron(input_count=2)
    history = []

    for epoch in range(epochs):
        total_loss = 0

        # Shuffle the four examples so the neuron does not see them in the
        # exact same order every epoch.
        shuffled_examples = AND_DATASET[:]
        random.shuffle(shuffled_examples)

        for inputs, target in shuffled_examples:
            total_loss += neuron.train_step(inputs, target, learning_rate)

        mean_loss = total_loss / len(AND_DATASET)
        history.append(mean_loss)

    return neuron, history


def decision_boundary_points(neuron):
    """
    Return two points on the neuron's decision boundary.

    The boundary is where the neuron is exactly undecided:

        sigmoid(z) = 0.5

    Since sigmoid(0) = 0.5, the separating line is:

        w1*x + w2*y + bias = 0

    Solving for y:

        y = -(w1*x + bias) / w2
    """
    w1, w2 = neuron.weights
    b = neuron.bias

    x_left = -0.25
    x_right = 1.25
    y_left = -(w1 * x_left + b) / w2
    y_right = -(w1 * x_right + b) / w2

    return [x_left, x_right], [y_left, y_right]


def plot_and_gate(neuron, output_path):
    """
    Draw the four AND input points and the neuron's separating line.

    This uses matplotlib only for visualization. The neuron itself, the forward
    pass, and the backpropagation loop above are pure Python.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "This script needs matplotlib only for the plot. Install it with:\n"
            "  .venv\\Scripts\\python.exe -m pip install matplotlib"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    zero_points = [inputs for inputs, target in AND_DATASET if target == 0]
    one_points = [inputs for inputs, target in AND_DATASET if target == 1]

    zero_x = [point[0] for point in zero_points]
    zero_y = [point[1] for point in zero_points]
    one_x = [point[0] for point in one_points]
    one_y = [point[1] for point in one_points]
    line_x, line_y = decision_boundary_points(neuron)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(zero_x, zero_y, s=140, label="target 0", color="#2f80ed")
    axis.scatter(one_x, one_y, s=140, label="target 1", color="#eb5757")
    axis.plot(line_x, line_y, color="#111111", linewidth=2, label="decision boundary")

    for inputs, target in AND_DATASET:
        output = neuron.forward(inputs)
        label = f"{inputs} -> target {target}, output {output:.3f}"
        axis.annotate(label, (inputs[0], inputs[1]), xytext=(8, 8), textcoords="offset points")

    axis.set_title("Single Neuron Learning AND")
    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.set_xlim(-0.25, 1.25)
    axis.set_ylim(-0.25, 1.25)
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def print_results(neuron, history, plot_path):
    """Print the learned neuron and verify all four AND examples."""
    print("Single neuron trained for AND")
    print("=============================")
    print(f"weights: w1={neuron.weights[0]:.4f}, w2={neuron.weights[1]:.4f}")
    print(f"bias:    b={neuron.bias:.4f}")
    print(f"loss:    start={history[0]:.6f}, end={history[-1]:.6f}")
    print()
    print("Truth table:")

    for inputs, target in AND_DATASET:
        output = neuron.forward(inputs)
        prediction = neuron.predict(inputs)
        print(
            f"  {inputs[0]} AND {inputs[1]} -> "
            f"target={target}, output={output:.4f}, prediction={prediction}"
        )

    print()
    print("Decision boundary:")
    print(
        "  The line is w1*x1 + w2*x2 + b = 0, "
        "which is where sigmoid output equals 0.5."
    )
    print(f"  Plot saved to: {plot_path}")


def main():
    neuron, history = train_and_neuron()
    plot_path = Path(__file__).parent / "assets" / "plots" / "and_neuron_boundary.png"
    plot_and_gate(neuron, plot_path)
    print_results(neuron, history, plot_path)


if __name__ == "__main__":
    main()
