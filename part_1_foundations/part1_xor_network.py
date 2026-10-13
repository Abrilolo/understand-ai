"""
Part I - The Ground: solving XOR with a tiny neural network.

This script shows two versions of the XOR solution:

1. A hand-built network with fixed weights and biases.
2. A trained network that learns its weights with backpropagation.

The important lesson is that XOR cannot be solved by one straight line, so a
single neuron is not enough. A small hidden layer gives us more than one
decision boundary, which is enough to solve the problem.
"""

import math
from pathlib import Path


XOR_DATASET = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]


def sigmoid(value):
    """Convert any real number into a smooth value between 0 and 1."""
    return 1 / (1 + math.exp(-value))


def sigmoid_derivative(output):
    """
    Return the derivative of sigmoid using the neuron's output.

    Backpropagation needs this because it must know how sensitive the neuron is:
    if we change the weighted sum a little, how much does the output change?
    """
    return output * (1 - output)


def step(value):
    """
    Convert a weighted sum into a hard 0 or 1.

    The hand-built XOR network uses step neurons because the chosen weights and
    biases are exact threshold rules. The trained network uses sigmoid neurons
    because gradient descent needs a smooth derivative.
    """
    return 1 if value >= 0 else 0


def dot(values, weights):
    """Compute a plain-Python dot product."""
    return sum(value * weight for value, weight in zip(values, weights))


class FixedStepNeuron:
    """A neuron with manually chosen weights, bias, and a step activation."""

    def __init__(self, weights, bias, name):
        self.weights = weights
        self.bias = bias
        self.name = name

    def forward(self, inputs):
        weighted_sum = dot(inputs, self.weights) + self.bias
        return step(weighted_sum)


class FixedXORNetwork:
    """
    A hand-built XOR network.

    Hidden neuron 1:
        weights = [1, 1], bias = -0.5
        This behaves like OR.

    Hidden neuron 2:
        weights = [-1, -1], bias = 1.5
        This behaves like NAND.

    Output neuron:
        weights = [1, 1], bias = -1.5
        This behaves like AND over the hidden outputs.

    XOR can be written as:

        XOR(x1, x2) = OR(x1, x2) AND NAND(x1, x2)
    """

    def __init__(self):
        self.hidden_or = FixedStepNeuron([1, 1], -0.5, "OR hidden neuron")
        self.hidden_nand = FixedStepNeuron([-1, -1], 1.5, "NAND hidden neuron")
        self.output_and = FixedStepNeuron([1, 1], -1.5, "AND output neuron")

    def forward(self, inputs):
        hidden_outputs = [
            self.hidden_or.forward(inputs),
            self.hidden_nand.forward(inputs),
        ]
        output = self.output_and.forward(hidden_outputs)
        return hidden_outputs, output


class SigmoidNeuron:
    """A trainable sigmoid neuron."""

    def __init__(self, weights, bias, name):
        self.weights = weights[:]
        self.bias = bias
        self.name = name
        self.last_inputs = None
        self.last_output = None

    def forward(self, inputs):
        """Run the neuron and store values needed for backpropagation."""
        weighted_sum = dot(inputs, self.weights) + self.bias
        self.last_inputs = inputs
        self.last_output = sigmoid(weighted_sum)
        return self.last_output


class TrainedXORNetwork:
    """
    A tiny 2-2-1 neural network.

    It has:
    - 2 inputs: x1 and x2
    - 2 hidden neurons
    - 1 output neuron

    The hidden layer lets the network build two intermediate features. During
    training, backpropagation decides what those features should be.
    """

    def __init__(self):
        self.hidden_layer = [
            # Start near the hand-built OR rule.
            # The fixed step version uses weights=[1, 1], bias=-0.5.
            # Here we use the same shape with sigmoid neurons, then let
            # backpropagation tune the values.
            SigmoidNeuron(weights=[1, 1], bias=-0.5, name="hidden_or_like"),

            # Start near the hand-built NAND rule.
            # The fixed step version uses weights=[-1, -1], bias=1.5.
            SigmoidNeuron(weights=[-1, -1], bias=1.5, name="hidden_nand_like"),
        ]

        # Start near the output AND rule over the two hidden features.
        # The fixed step version uses weights=[1, 1], bias=-1.5.
        self.output_neuron = SigmoidNeuron(weights=[1, 1], bias=-1.5, name="output_and_like")

    def forward(self, inputs):
        """Run inputs through hidden neurons, then through the output neuron."""
        hidden_outputs = [neuron.forward(inputs) for neuron in self.hidden_layer]
        output = self.output_neuron.forward(hidden_outputs)
        return hidden_outputs, output

    def predict(self, inputs):
        """Turn the smooth output into a hard XOR prediction."""
        _, output = self.forward(inputs)
        return 1 if output >= 0.5 else 0

    def train_step(self, inputs, target, learning_rate):
        """
        Run one complete backpropagation update.

        Forward pass:
            1. Compute both hidden neuron outputs.
            2. Compute the final network output.

        Backward pass:
            1. Measure the output error.
            2. Compute the output neuron's delta.
            3. Send that delta backward into each hidden neuron.
            4. Update output weights, hidden weights, and all biases.
        """
        hidden_outputs, output = self.forward(inputs)

        # Squared error keeps the math visible for this beginner version.
        loss = 0.5 * (target - output) ** 2

        # This is d_loss/d_output times d_output/d_weighted_sum.
        output_error = output - target
        output_delta = output_error * sigmoid_derivative(output)

        # Save the old output weights because hidden deltas must be calculated
        # using the weights that produced the current error.
        old_output_weights = self.output_neuron.weights[:]

        # Update the output neuron. Each output weight receives one hidden
        # neuron's activation as its input.
        for index, hidden_output in enumerate(hidden_outputs):
            gradient = output_delta * hidden_output
            self.output_neuron.weights[index] -= learning_rate * gradient
        self.output_neuron.bias -= learning_rate * output_delta

        # Push the output error backward into the hidden layer.
        for hidden_index, hidden_neuron in enumerate(self.hidden_layer):
            hidden_output = hidden_neuron.last_output
            hidden_error = output_delta * old_output_weights[hidden_index]
            hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

            # Update each hidden weight. Each hidden neuron receives x1 and x2
            # directly as inputs.
            for input_index, input_value in enumerate(inputs):
                gradient = hidden_delta * input_value
                hidden_neuron.weights[input_index] -= learning_rate * gradient
            hidden_neuron.bias -= learning_rate * hidden_delta

        return loss


def train_xor_network(epochs=50000, learning_rate=0.8):
    """Train the sigmoid network until it solves XOR."""
    network = TrainedXORNetwork()
    history = []

    for _ in range(epochs):
        total_loss = 0

        for inputs, target in XOR_DATASET:
            total_loss += network.train_step(inputs, target, learning_rate)

        history.append(total_loss / len(XOR_DATASET))

    return network, history


def print_fixed_network_results(network):
    """Print the truth table for the hand-built XOR network."""
    print("Fixed XOR network")
    print("=================")
    print("hidden OR:   weights=[1, 1],   bias=-0.5")
    print("hidden NAND: weights=[-1, -1], bias=1.5")
    print("output AND:  weights=[1, 1],   bias=-1.5")
    print()
    print("Truth table:")

    for inputs, target in XOR_DATASET:
        hidden_outputs, output = network.forward(inputs)
        print(
            f"  {inputs[0]} XOR {inputs[1]} -> "
            f"OR={hidden_outputs[0]}, NAND={hidden_outputs[1]}, "
            f"target={target}, prediction={output}"
        )


def print_trained_network_results(network, history):
    """Print learned weights and the truth table for the trained network."""
    print()
    print("Trained XOR network")
    print("===================")
    print(f"loss: start={history[0]:.6f}, end={history[-1]:.6f}")
    print()

    for neuron in network.hidden_layer:
        print(
            f"{neuron.name}: "
            f"weights=[{neuron.weights[0]:.4f}, {neuron.weights[1]:.4f}], "
            f"bias={neuron.bias:.4f}"
        )

    output = network.output_neuron
    print(
        f"{output.name}: "
        f"weights=[{output.weights[0]:.4f}, {output.weights[1]:.4f}], "
        f"bias={output.bias:.4f}"
    )
    print()
    print("Truth table:")

    for inputs, target in XOR_DATASET:
        hidden_outputs, output_value = network.forward(inputs)
        prediction = network.predict(inputs)
        print(
            f"  {inputs[0]} XOR {inputs[1]} -> "
            f"hidden=[{hidden_outputs[0]:.4f}, {hidden_outputs[1]:.4f}], "
            f"target={target}, output={output_value:.4f}, prediction={prediction}"
        )


def load_matplotlib():
    """Load matplotlib only for plots; all neural-network math is pure Python."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "This script needs matplotlib only for the plots. Install it with:\n"
            "  .venv\\Scripts\\python.exe -m pip install matplotlib"
        ) from exc

    return plt


def line_points(weights, bias):
    """Return two points for the line w1*x + w2*y + b = 0."""
    w1, w2 = weights
    x_left = -0.25
    x_right = 1.25
    y_left = -(w1 * x_left + bias) / w2
    y_right = -(w1 * x_right + bias) / w2
    return [x_left, x_right], [y_left, y_right]


def plot_fixed_network(output_path):
    """Plot the two hand-built hidden decision boundaries."""
    plt = load_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 6))
    add_xor_points(axis)

    or_x, or_y = line_points([1, 1], -0.5)
    nand_x, nand_y = line_points([-1, -1], 1.5)
    axis.plot(or_x, or_y, color="#2f80ed", linewidth=2, label="OR boundary")
    axis.plot(nand_x, nand_y, color="#27ae60", linewidth=2, label="NAND boundary")

    axis.set_title("Fixed XOR Network Hidden Boundaries")
    format_xor_axis(axis)
    axis.legend(loc="lower center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_trained_network(network, output_path):
    """Plot only the trained network's decision boundary and the XOR points."""
    plt = load_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_min = -0.25
    x_max = 1.25
    y_min = -0.25
    y_max = 1.25
    grid_size = 80
    xs = [
        x_min + index * (x_max - x_min) / (grid_size - 1)
        for index in range(grid_size)
    ]
    ys = [
        y_min + index * (y_max - y_min) / (grid_size - 1)
        for index in range(grid_size)
    ]
    z_values = []

    for y_value in ys:
        row = []
        for x_value in xs:
            _, output = network.forward([x_value, y_value])
            row.append(output)
        z_values.append(row)

    figure, axis = plt.subplots(figsize=(6, 6))

    # Draw only the classification boundary. This keeps the plot visually close
    # to the simpler AND plot: points on a square plus the learned separator.
    axis.contour(xs, ys, z_values, levels=[0.5], colors="#111111", linewidths=2)
    axis.plot([], [], color="#111111", linewidth=2, label="output = 0.5")
    add_xor_points(axis)

    axis.set_title("Trained XOR Network Decision Region")
    format_xor_axis(axis)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def add_xor_points(axis):
    """Draw the four XOR training points."""
    zero_points = [inputs for inputs, target in XOR_DATASET if target == 0]
    one_points = [inputs for inputs, target in XOR_DATASET if target == 1]

    axis.scatter(
        [point[0] for point in zero_points],
        [point[1] for point in zero_points],
        s=150,
        color="#2f80ed",
        edgecolor="#111111",
        label="target 0",
    )
    axis.scatter(
        [point[0] for point in one_points],
        [point[1] for point in one_points],
        s=150,
        color="#eb5757",
        edgecolor="#111111",
        label="target 1",
    )


def format_xor_axis(axis):
    """Keep all XOR plots on the same input square."""
    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.set_xlim(-0.25, 1.25)
    axis.set_ylim(-0.25, 1.25)
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.grid(True, alpha=0.25)


def main():
    fixed_network = FixedXORNetwork()
    trained_network, history = train_xor_network()

    fixed_plot_path = Path(__file__).parent / "assets" / "plots" / "xor_fixed_network.png"
    trained_plot_path = Path(__file__).parent / "assets" / "plots" / "xor_trained_network.png"

    print_fixed_network_results(fixed_network)
    print_trained_network_results(trained_network, history)

    plot_fixed_network(fixed_plot_path)
    plot_trained_network(trained_network, trained_plot_path)

    print()
    print(f"Fixed network plot saved to: {fixed_plot_path}")
    print(f"Trained network plot saved to: {trained_plot_path}")


if __name__ == "__main__":
    main()
