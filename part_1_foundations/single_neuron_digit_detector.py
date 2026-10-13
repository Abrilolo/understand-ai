#Part I - The Ground: a neuron, then a network, in pure Python.


import math
import random


# ---------------------------------------------------------------------------
# Activation function
# ---------------------------------------------------------------------------
# Sigmoid squashes any real number into the range (0, 1). We use it because
# it is smooth and differentiable everywhere, which is exactly what
# backpropagation needs (it has to compute a derivative at every neuron).

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

#ReLu
 
def sigmoid_derivative(activated_value):
    # Nice property of sigmoid: if s = sigmoid(z), then sigmoid'(z) = s * (1 - s).
    # We already have the activated output stored on the neuron, so we can
    # compute the derivative from it directly without touching z again.
    return activated_value * (1 - activated_value)


# ---------------------------------------------------------------------------
# Part A: a single neuron
# ---------------------------------------------------------------------------

class Neuron:
    """One sigmoid neuron: weighted sum of inputs, plus a bias, through sigmoid."""

    def __init__(self, num_inputs):
        # Small random starting weights/bias, roughly following Nielsen's setup.
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        # Filled in on every forward() call, needed later by backward().
        self.last_inputs = None
        self.last_output = None

    def forward(self, inputs):
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.last_inputs = inputs
        self.last_output = sigmoid(z)
        return self.last_output


def train_single_neuron(neuron, dataset, epochs, learning_rate):
    """
    Manual gradient descent for one neuron, spelled out step by step.

    dataset: list of (inputs, target) pairs.
    This is the same math the network below uses, just without any of the
    layer-to-layer bookkeeping, so it's the easiest place to see it clearly.
    """
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, target in dataset:
            output = neuron.forward(inputs)

            error = target - output
            total_loss += error ** 2

            # How much should z change to reduce the error? This is the
            # "delta": error scaled by how sensitive sigmoid is at this point.
            delta = error * sigmoid_derivative(output)

            # Nudge each weight in the direction that reduces the error.
            # d(loss)/d(weight_i) = delta * input_i, so we step against it.
            for i in range(len(neuron.weights)):
                neuron.weights[i] += learning_rate * delta * inputs[i]
            neuron.bias += learning_rate * delta

        if epoch % (epochs // 10 or 1) == 0:
            print(f"  epoch {epoch:4d}  mse={total_loss / len(dataset):.4f}")


# ---------------------------------------------------------------------------
# Part B: a small multi-layer network
# ---------------------------------------------------------------------------

class Layer:
    """A row of neurons, all reading the same inputs."""

    def __init__(self, num_inputs, num_neurons):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def forward(self, inputs):
        return [neuron.forward(inputs) for neuron in self.neurons]

    
class NeuralNetwork:
    """A stack of Layers, e.g. NeuralNetwork([2, 4, 1]) = 2 inputs -> 4 hidden -> 1 output."""

    def __init__(self, layer_sizes):
        self.layers = [
            Layer(layer_sizes[i - 1], layer_sizes[i])
            for i in range(1, len(layer_sizes))
        ]

    def forward(self, inputs):
        activations = inputs
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def backward(self, targets, learning_rate):
        """
        One backpropagation update, after a forward() call has already run.

        Two passes:
          1. Right to left: compute each neuron's "delta" (how much its own
             weighted sum should change to reduce the loss).
          2. Apply the weight/bias updates using those deltas.
        These are kept separate because a hidden neuron's delta depends on
        the *current* (pre-update) weights of the next layer, so we can't
        update weights and compute deltas in the same pass.
        """
        # --- pass 1: deltas, output layer first ---
        deltas_per_layer = [None] * len(self.layers)

        output_layer = self.layers[-1]
        output_deltas = []
        for neuron, target in zip(output_layer.neurons, targets):
            error = target - neuron.last_output
            output_deltas.append(error * sigmoid_derivative(neuron.last_output))
        deltas_per_layer[-1] = output_deltas

        # walk backwards through the hidden layers
        for l in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[l]
            next_layer = self.layers[l + 1]
            next_deltas = deltas_per_layer[l + 1]

            layer_deltas = []
            for i, neuron in enumerate(layer.neurons):
                # This neuron's error is the sum of every next-layer delta,
                # weighted by how strongly this neuron feeds into that neuron.
                # This is the chain rule: error flows backward through the
                # same weights it flowed forward through.
                error = sum(
                    next_deltas[j] * next_layer.neurons[j].weights[i]
                    for j in range(len(next_layer.neurons))
                )
                layer_deltas.append(error * sigmoid_derivative(neuron.last_output))
            deltas_per_layer[l] = layer_deltas

        # --- pass 2: apply the updates ---
        for layer, layer_deltas in zip(self.layers, deltas_per_layer):
            for neuron, delta in zip(layer.neurons, layer_deltas):
                for i in range(len(neuron.weights)):
                    neuron.weights[i] += learning_rate * delta * neuron.last_inputs[i]
                neuron.bias += learning_rate * delta

    def train(self, dataset, epochs, learning_rate):
        """dataset: list of (inputs, targets) pairs. targets is a list too."""
        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, targets in dataset:
                outputs = self.forward(inputs)
                total_loss += sum((t - o) ** 2 for t, o in zip(targets, outputs))
                self.backward(targets, learning_rate)

            if epoch % (epochs // 10 or 1) == 0:
                print(f"  epoch {epoch:4d}  mse={total_loss / len(dataset):.4f}")


def load_digits_dataset():
    try:
        from sklearn.datasets import load_digits
    except ImportError as exc:
        raise SystemExit(
            "This demo needs scikit-learn. Install it with:\n"
            "  .venv\\Scripts\\python.exe -m pip install scikit-learn"
        ) from exc

    return load_digits()


def flatten_digit_image(image):
    return [pixel for row in image for pixel in row]


def binarize_pixels(pixels):
    # The sklearn digits dataset stores pixel intensity from 0 to 16.
    return [1 if pixel >= 8 else 0 for pixel in pixels]


def pixel_to_shade(pixel):
    shades = " .:-=+*#%@"
    index = round((pixel / 16) * (len(shades) - 1))
    return shades[index]


def print_digit(image):
    for row in image:
        pixels = "".join(pixel_to_shade(pixel) * 2 for pixel in row)
        print(f"    |{pixels}|")


def first_sample_per_digit(digits):
    samples = {}

    for image, target in zip(digits.images, digits.target):
        digit = int(target)
        if digit not in samples:
            samples[digit] = image

        if len(samples) == 10:
            return samples

    return samples


def make_digit_detector(image):
    """
    A very basic hand-built detector for one digit.

    Pixels that should be on get positive weights.
    Pixels that should be off get negative weights.
    The bias acts like a threshold, so the neuron fires when enough pixels match.
    """
    expected_pixels = binarize_pixels(flatten_digit_image(image))
    detector = Neuron(num_inputs=len(expected_pixels))
    detector.weights = [2 if pixel == 1 else -2 for pixel in expected_pixels]
    detector.bias = -(2 * sum(expected_pixels) - 1)
    return detector


def detect_digit(detectors, image):
    inputs = binarize_pixels(flatten_digit_image(image))
    scores = [
        (digit, detector.forward(inputs))
        for digit, detector in detectors.items()
    ]
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[0][0], scores


def main():
    digits = load_digits_dataset()
    reference_images = first_sample_per_digit(digits)
    detectors = {
        digit: make_digit_detector(image)
        for digit, image in reference_images.items()
    }

    print("Basic digit detector with sklearn's digits dataset")
    print("No training yet: one neuron per digit, built from one reference image.")
    print(f"Dataset size: {len(digits.images)} images, each one is 8 x 8 pixels.")
    print()

    for index in range(5):
        image = digits.images[index]
        real_digit = int(digits.target[index])
        predicted_digit, scores = detect_digit(detectors, image)
        top_scores = ", ".join(
            f"{digit}: {score:.3f}"
            for digit, score in scores[:3]
        )

        print(f"Sample #{index} | real digit: {real_digit}")
        print_digit(image)
        print(f"  prediction: {predicted_digit}")
        print(f"  top scores: {top_scores}")
        print()

    total = min(100, len(digits.images))
    correct = 0
    for index in range(total):
        predicted_digit, _ = detect_digit(detectors, digits.images[index])
        real_digit = int(digits.target[index])
        if predicted_digit == real_digit:
            correct += 1

    print(f"Rudimentary accuracy on the first {total} samples: {correct}/{total}")


if __name__ == "__main__":
    main()
