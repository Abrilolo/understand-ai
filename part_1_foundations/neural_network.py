#Part I - The Ground: a neuron, then a network, in pure Python.


import argparse
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


def normalize_pixels(pixels):
    # The sklearn digits dataset stores pixel intensity from 0 to 16.
    return [pixel / 16 for pixel in pixels]


def pixel_to_shade(pixel):
    shades = " .:-=+*#%@"
    index = round((pixel / 16) * (len(shades) - 1))
    return shades[index]


def print_digit(image):
    for row in image:
        pixels = "".join(pixel_to_shade(pixel) * 2 for pixel in row)
        print(f"    |{pixels}|")


STROKE_REGIONS = {
    "top": [(row, col) for row in range(0, 2) for col in range(2, 6)],
    "middle": [(row, col) for row in range(3, 5) for col in range(2, 6)],
    "bottom": [(row, col) for row in range(6, 8) for col in range(2, 6)],
    "upper_left": [(row, col) for row in range(1, 4) for col in range(1, 3)],
    "lower_left": [(row, col) for row in range(4, 7) for col in range(1, 3)],
    "upper_right": [(row, col) for row in range(1, 4) for col in range(5, 7)],
    "lower_right": [(row, col) for row in range(4, 7) for col in range(5, 7)],
    "center_vertical": [(row, col) for row in range(1, 7) for col in range(3, 5)],
}


DIGIT_STROKES = {
    0: ["top", "bottom", "upper_left", "lower_left", "upper_right", "lower_right"],
    1: ["center_vertical", "lower_right"],
    2: ["top", "middle", "bottom", "upper_right", "lower_left"],
    3: ["top", "middle", "bottom", "upper_right", "lower_right"],
    4: ["middle", "upper_left", "upper_right", "lower_right"],
    5: ["top", "middle", "bottom", "upper_left", "lower_right"],
    6: ["top", "middle", "bottom", "upper_left", "lower_left", "lower_right"],
    7: ["top", "upper_right", "center_vertical"],
    8: ["top", "middle", "bottom", "upper_left", "lower_left", "upper_right", "lower_right"],
    9: ["top", "middle", "bottom", "upper_left", "upper_right", "lower_right"],
}


def pixel_index(row, col):
    return row * 8 + col


def make_stroke_detector(region, threshold=0.32, strength=10):
    """
    First-layer neuron.

    It looks at a small region of the 8x8 image and activates when that region
    has enough dark pixels to look like a stroke or contour.
    """
    detector = Neuron(num_inputs=64)
    detector.weights = [0] * 64

    for row, col in region:
        detector.weights[pixel_index(row, col)] = strength / len(region)

    detector.bias = -strength * threshold
    return detector


def make_stroke_layer():
    return {
        name: make_stroke_detector(region)
        for name, region in STROKE_REGIONS.items()
    }


def make_digit_neuron(expected_strokes, stroke_names):
    """
    Second-layer neuron.

    It does not look at pixels directly. It looks at which first-layer stroke
    detectors fired and combines them into a rough digit guess.
    """
    neuron = Neuron(num_inputs=len(stroke_names))
    neuron.weights = [
        2.2 if name in expected_strokes else -1.0
        for name in stroke_names
    ]
    neuron.bias = -(1.6 * len(expected_strokes))
    return neuron


def make_digit_layer(stroke_names):
    return {
        digit: make_digit_neuron(expected_strokes, stroke_names)
        for digit, expected_strokes in DIGIT_STROKES.items()
    }


def forward_stroke_layer(stroke_layer, image):
    inputs = normalize_pixels(flatten_digit_image(image))
    return {
        name: detector.forward(inputs)
        for name, detector in stroke_layer.items()
    }


def detect_digit(stroke_layer, digit_layer, stroke_names, image):
    stroke_outputs = forward_stroke_layer(stroke_layer, image)
    stroke_activations = [stroke_outputs[name] for name in stroke_names]
    digit_scores = [
        (digit, neuron.forward(stroke_activations))
        for digit, neuron in digit_layer.items()
    ]
    digit_scores.sort(key=lambda item: item[1], reverse=True)
    return digit_scores[0][0], digit_scores, stroke_outputs


def format_top_strokes(stroke_outputs, limit=5):
    top_strokes = sorted(
        stroke_outputs.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return ", ".join(
        f"{name}: {activation:.2f}"
        for name, activation in top_strokes[:limit]
    )


def print_stroke_region(image, stroke_name):
    region = set(STROKE_REGIONS[stroke_name])

    for row in range(8):
        pixels = []
        for col in range(8):
            if (row, col) in region:
                pixels.append(pixel_to_shade(image[row][col]) * 2)
            else:
                pixels.append("  ")
        print(f"      |{''.join(pixels)}|")


def print_first_layer_detections(image, stroke_outputs, limit=3):
    top_strokes = sorted(
        stroke_outputs.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    print("  layer 1 detects small strokes:")
    for stroke_name, activation in top_strokes[:limit]:
        print(f"    neuron '{stroke_name}' activation: {activation:.2f}")
        print_stroke_region(image, stroke_name)


def print_second_layer_detections(scores, stroke_outputs, limit=3, active_threshold=0.65):
    active_strokes = {
        stroke_name
        for stroke_name, activation in stroke_outputs.items()
        if activation >= active_threshold
    }

    print("  layer 2 combines strokes into digit guesses:")
    for digit, score in scores[:limit]:
        expected = DIGIT_STROKES[digit]
        detected = [stroke for stroke in expected if stroke in active_strokes]
        missing = [stroke for stroke in expected if stroke not in active_strokes]

        print(f"    digit {digit} score: {score:.3f}")
        print(f"      expects: {', '.join(expected)}")
        print(f"      detected: {', '.join(detected) or 'none'}")
        print(f"      missing: {', '.join(missing) or 'none'}")


def activation_bar(activation, width=20):
    filled = round(activation * width)
    return "#" * filled + "." * (width - filled)


def print_output_layer(scores, active_threshold=0.5):
    winning_digit, winning_activation = scores[0]
    active_outputs = [
        digit
        for digit, activation in scores
        if activation >= active_threshold
    ]

    print("  output layer neurons:")
    for digit, activation in sorted(scores):
        status = "ON " if activation >= active_threshold else "off"
        winner = " <- strongest output / prediction" if digit == winning_digit else ""
        print(
            f"    neuron for digit {digit}: {activation:.3f} "
            f"[{activation_bar(activation)}] {status}{winner}"
        )

    if active_outputs:
        print(f"  output neurons above {active_threshold:.2f}: {active_outputs}")
    else:
        print(
            f"  no output neuron passed {active_threshold:.2f}; "
            f"prediction uses the strongest one: {winning_digit} "
            f"({winning_activation:.3f})"
        )


def build_digit_detector():
    stroke_layer = make_stroke_layer()
    stroke_names = list(stroke_layer.keys())
    digit_layer = make_digit_layer(stroke_names)
    return stroke_layer, stroke_names, digit_layer


def analyze_sample(digits, detector, index, show_layers=True):
    stroke_layer, stroke_names, digit_layer = detector
    image = digits.images[index]
    real_digit = int(digits.target[index])
    predicted_digit, scores, stroke_outputs = detect_digit(
        stroke_layer,
        digit_layer,
        stroke_names,
        image,
    )
    top_scores = ", ".join(
        f"{digit}: {score:.3f}"
        for digit, score in scores[:3]
    )

    print(f"Sample #{index} | real digit: {real_digit}")
    print("  original image:")
    print_digit(image)
    print(f"  first layer strokes: {format_top_strokes(stroke_outputs)}")

    if show_layers:
        print_first_layer_detections(image, stroke_outputs)

    print(f"  prediction: output neuron {predicted_digit}")
    print(f"  strongest output scores: {top_scores}")
    print_output_layer(scores)

    if show_layers:
        print_second_layer_detections(scores, stroke_outputs)

    print()
    return predicted_digit == real_digit


def matching_indexes(digits, wanted_digit):
    return [
        index
        for index, target in enumerate(digits.target)
        if int(target) == wanted_digit
    ]


def evaluate_accuracy(digits, detector, limit):
    stroke_layer, stroke_names, digit_layer = detector
    total = min(limit, len(digits.images))
    correct = 0

    for index in range(total):
        predicted_digit, _, _ = detect_digit(
            stroke_layer,
            digit_layer,
            stroke_names,
            digits.images[index],
        )
        real_digit = int(digits.target[index])
        if predicted_digit == real_digit:
            correct += 1

    return correct, total


def run_demo(digits, detector, limit):
    print("Hierarchical digit detector with sklearn's digits dataset")
    print("No training yet: pixels activate stroke neurons, then strokes activate digit neurons.")
    print(f"Dataset size: {len(digits.images)} images, each one is 8 x 8 pixels.")
    print()

    for index in range(min(limit, len(digits.images))):
        analyze_sample(digits, detector, index)

    correct, total = evaluate_accuracy(digits, detector, 100)
    print(f"Rudimentary accuracy on the first {total} samples: {correct}/{total}")


def run_digit_examples(digits, detector, wanted_digit, limit):
    indexes = matching_indexes(digits, wanted_digit)
    selected_indexes = indexes[:limit]

    print(f"Digit label requested: {wanted_digit}")
    print(f"Found {len(indexes)} images with that label.")
    print(f"Showing {len(selected_indexes)} example(s).")
    print()

    for index in selected_indexes:
        analyze_sample(digits, detector, index)


def run_single_index(digits, detector, index):
    if index < 0 or index >= len(digits.images):
        raise SystemExit(f"Index must be between 0 and {len(digits.images) - 1}")

    analyze_sample(digits, detector, index)


def parse_digit(value):
    digit = int(value)
    if digit < 0 or digit > 9:
        raise argparse.ArgumentTypeError("digit must be from 0 to 9")
    return digit


def ask_int(prompt, minimum=None, maximum=None, default=None):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"{prompt}{suffix}: ").strip()

        if value == "" and default is not None:
            return default

        try:
            number = int(value)
        except ValueError:
            print("Please type a whole number.")
            continue

        if minimum is not None and number < minimum:
            print(f"Please type a number >= {minimum}.")
            continue

        if maximum is not None and number > maximum:
            print(f"Please type a number <= {maximum}.")
            continue

        return number


def run_menu(digits, detector):
    while True:
        print()
        print("Digit detector menu")
        print("  1. Analyze one exact dataset index")
        print("  2. Analyze examples by digit label")
        print("  3. Run a short guided demo")
        print("  4. Measure rudimentary accuracy")
        print("  5. Exit")

        choice = ask_int("Choose an option", 1, 5)
        print()

        if choice == 1:
            index = ask_int("Dataset index", 0, len(digits.images) - 1, 0)
            print()
            run_single_index(digits, detector, index)
        elif choice == 2:
            digit = ask_int("Digit label", 0, 9, 5)
            limit = ask_int("How many examples to show", 1, 20, 3)
            print()
            run_digit_examples(digits, detector, digit, limit)
        elif choice == 3:
            limit = ask_int("How many starting samples to show", 1, 20, 3)
            print()
            run_demo(digits, detector, limit)
        elif choice == 4:
            limit = ask_int("How many samples to test", 1, len(digits.images), 100)
            correct, total = evaluate_accuracy(digits, detector, limit)
            print(f"Rudimentary accuracy on the first {total} samples: {correct}/{total}")
        else:
            print("Bye.")
            return


def main():
    parser = argparse.ArgumentParser(
        description="Explore a tiny, hand-built hierarchical digit detector."
    )
    parser.add_argument(
        "--index",
        type=int,
        help="analyze one exact sklearn dataset index",
    )
    parser.add_argument(
        "--digit",
        type=parse_digit,
        help="show examples whose label is this digit, from 0 to 9",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="number of examples to show for --digit or --demo",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="show a guided demo from the start of the dataset",
    )
    parser.add_argument(
        "--accuracy",
        type=int,
        help="only measure accuracy over the first N samples",
    )
    parser.add_argument(
        "--menu",
        action="store_true",
        help="open the interactive menu",
    )
    args = parser.parse_args()

    digits = load_digits_dataset()
    detector = build_digit_detector()

    if args.index is not None:
        run_single_index(digits, detector, args.index)
    elif args.digit is not None:
        run_digit_examples(digits, detector, args.digit, args.limit)
    elif args.demo:
        run_demo(digits, detector, args.limit)
    elif args.accuracy is not None:
        correct, total = evaluate_accuracy(digits, detector, args.accuracy)
        print(f"Rudimentary accuracy on the first {total} samples: {correct}/{total}")
    else:
        run_menu(digits, detector)


if __name__ == "__main__":
    main()