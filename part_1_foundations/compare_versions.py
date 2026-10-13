import argparse
import sys
import time
from dataclasses import dataclass

from neural_network import (
    DIGIT_STROKES,
    Neuron,
    build_digit_detector,
    detect_digit,
    format_top_strokes,
    load_digits_dataset,
    pixel_to_shade,
    print_digit,
    print_first_layer_detections,
    print_output_layer,
    print_second_layer_detections,
)


# Change experiments here.
# kind:
#   baseline = the same hand-built stroke detector used in neural_network.py
#   strokes = segment-style regions, like top/middle/bottom/left/right
#   patches = more neurons: a grid of small patch detectors
VERSION_CONFIGS = [
    {
        "name": "baseline from neural_network.py",
        "kind": "baseline",
    },
    {
        "name": "strokes 16x16",
        "kind": "strokes",
        "upscale": 2,
        "threshold": 0.32,
        "strength": 10,
        "positive_weight": 2.2,
        "negative_weight": -1.0,
        "bias_scale": 1.6,
    },
    {
        "name": "patches 4x4 on 8x8",
        "kind": "patches",
        "grid": 4,
        "upscale": 1,
        "threshold": 0.18,
        "strength": 8,
        "prototype_bias": 0.55,
    },
    {
        "name": "patches 4x4 on 16x16",
        "kind": "patches",
        "grid": 4,
        "upscale": 2,
        "threshold": 0.18,
        "strength": 8,
        "prototype_bias": 0.55,
    },
    {
        "name": "patches 8x8 on 16x16",
        "kind": "patches",
        "grid": 8,
        "upscale": 2,
        "threshold": 0.48,
        "strength": 8,
        "prototype_bias": 0.75,
    },
]


def normalize_image(image):
    return [pixel / 16 for row in image for pixel in row]


def upscale_image(image, factor):
    if factor == 1:
        return image

    expanded = []
    for row in image:
        expanded_row = []
        for pixel in row:
            expanded_row.extend([pixel] * factor)

        for _ in range(factor):
            expanded.append(expanded_row[:])

    return expanded


def pixel_index(row, col, size):
    return row * size + col


def rectangle_region(size, top, bottom, left, right):
    row_start = max(0, round(size * top))
    row_end = min(size, round(size * bottom))
    col_start = max(0, round(size * left))
    col_end = min(size, round(size * right))

    return [
        (row, col)
        for row in range(row_start, row_end)
        for col in range(col_start, col_end)
    ]


def make_segment_regions(size):
    return {
        "top": rectangle_region(size, 0.00, 0.25, 0.25, 0.75),
        "middle": rectangle_region(size, 0.38, 0.62, 0.25, 0.75),
        "bottom": rectangle_region(size, 0.75, 1.00, 0.25, 0.75),
        "upper_left": rectangle_region(size, 0.12, 0.50, 0.12, 0.38),
        "lower_left": rectangle_region(size, 0.50, 0.88, 0.12, 0.38),
        "upper_right": rectangle_region(size, 0.12, 0.50, 0.62, 0.88),
        "lower_right": rectangle_region(size, 0.50, 0.88, 0.62, 0.88),
        "center_vertical": rectangle_region(size, 0.12, 0.88, 0.38, 0.62),
    }


def make_patch_regions(size, grid_size):
    regions = {}
    for patch_row in range(grid_size):
        for patch_col in range(grid_size):
            name = f"patch_{patch_row}_{patch_col}"
            regions[name] = rectangle_region(
                size,
                patch_row / grid_size,
                (patch_row + 1) / grid_size,
                patch_col / grid_size,
                (patch_col + 1) / grid_size,
            )

    return regions


def make_region_detector(region, image_size, threshold, strength):
    detector = Neuron(num_inputs=image_size * image_size)
    detector.weights = [0] * (image_size * image_size)

    for row, col in region:
        detector.weights[pixel_index(row, col, image_size)] = strength / len(region)

    detector.bias = -strength * threshold
    return detector


def make_region_layer(regions, image_size, threshold, strength):
    return {
        name: make_region_detector(region, image_size, threshold, strength)
        for name, region in regions.items()
    }


def forward_region_layer(region_layer, image):
    inputs = normalize_image(image)
    return {
        name: detector.forward(inputs)
        for name, detector in region_layer.items()
    }


def first_sample_per_digit(digits):
    samples = {}
    for image, target in zip(digits.images, digits.target):
        digit = int(target)
        if digit not in samples:
            samples[digit] = image
        if len(samples) == 10:
            return samples

    return samples


def make_hand_digit_layer(region_names, positive_weight, negative_weight, bias_scale):
    layer = {}
    for digit, expected_regions in DIGIT_STROKES.items():
        neuron = Neuron(num_inputs=len(region_names))
        neuron.weights = [
            positive_weight if name in expected_regions else negative_weight
            for name in region_names
        ]
        neuron.bias = -(bias_scale * len(expected_regions))
        layer[digit] = neuron

    return layer


def make_prototype_digit_layer(
    digits,
    region_layer,
    region_names,
    upscale_factor,
    prototype_bias,
):
    samples = first_sample_per_digit(digits)
    layer = {}

    for digit, image in samples.items():
        image = upscale_image(image, upscale_factor)
        outputs = forward_region_layer(region_layer, image)
        prototype = [outputs[name] for name in region_names]

        neuron = Neuron(num_inputs=len(region_names))
        neuron.weights = [(activation * 2) - 1 for activation in prototype]
        neuron.bias = -prototype_bias * sum(max(weight, 0) for weight in neuron.weights)
        layer[digit] = neuron

    return layer


@dataclass
class PredictionDetails:
    prediction: int
    scores: list
    layer_outputs: dict
    shown_image: list


@dataclass
class VersionResult:
    version_id: int
    name: str
    correct: int
    total: int
    seconds: float
    per_digit: dict

    @property
    def accuracy(self):
        return self.correct / self.total if self.total else 0


class BaselineVersion:
    def __init__(self, config):
        self.name = config["name"]
        self.kind = config["kind"]
        self.upscale_factor = 1
        self.layer_label = "stroke"
        self.detector = build_digit_detector()

    def predict_details(self, image):
        stroke_layer, stroke_names, digit_layer = self.detector
        prediction, scores, stroke_outputs = detect_digit(
            stroke_layer,
            digit_layer,
            stroke_names,
            image,
        )
        return PredictionDetails(prediction, scores, stroke_outputs, image)

    def predict(self, image):
        return self.predict_details(image).prediction

    def print_layers(self, details):
        print(f"  first layer strokes: {format_top_strokes(details.layer_outputs)}")
        print_first_layer_detections(details.shown_image, details.layer_outputs)
        print(f"  prediction: output neuron {details.prediction}")
        print_output_layer(details.scores)
        print_second_layer_detections(details.scores, details.layer_outputs)


class ConfigurableVersion:
    def __init__(self, config, digits):
        self.name = config["name"]
        self.kind = config["kind"]
        self.upscale_factor = config.get("upscale", 1)
        self.image_size = 8 * self.upscale_factor
        self.layer_label = "stroke" if self.kind == "strokes" else "patch"
        self.layer_plural = "strokes" if self.kind == "strokes" else "patches"

        if self.kind == "strokes":
            self.regions = make_segment_regions(self.image_size)
            self.digit_layer = make_hand_digit_layer(
                list(self.regions.keys()),
                config.get("positive_weight", 2.2),
                config.get("negative_weight", -1.0),
                config.get("bias_scale", 1.6),
            )
        else:
            self.regions = make_patch_regions(self.image_size, config.get("grid", 4))
            self.digit_layer = None

        self.region_names = list(self.regions.keys())
        self.region_layer = make_region_layer(
            self.regions,
            self.image_size,
            config.get("threshold", 0.32),
            config.get("strength", 10),
        )

        if self.kind == "patches":
            self.digit_layer = make_prototype_digit_layer(
                digits,
                self.region_layer,
                self.region_names,
                self.upscale_factor,
                config.get("prototype_bias", 0.55),
            )

    def predict_details(self, image):
        shown_image = upscale_image(image, self.upscale_factor)
        region_outputs = forward_region_layer(self.region_layer, shown_image)
        activations = [region_outputs[name] for name in self.region_names]
        scores = [
            (digit, neuron.forward(activations))
            for digit, neuron in self.digit_layer.items()
        ]
        scores.sort(key=lambda item: item[1], reverse=True)
        return PredictionDetails(scores[0][0], scores, region_outputs, shown_image)

    def predict(self, image):
        return self.predict_details(image).prediction

    def print_layers(self, details):
        print(f"  first layer {self.layer_plural}: {format_top_items(details.layer_outputs)}")
        print_top_regions(details.shown_image, self.regions, details.layer_outputs, self.layer_label)
        print(f"  prediction: output neuron {details.prediction}")
        print_output_layer(details.scores)
        print_output_matches(details.scores, details.layer_outputs, self.region_names, self.digit_layer)


def build_versions_from_configs(digits, configs):
    versions = []
    for config in configs:
        if config["kind"] == "baseline":
            versions.append(BaselineVersion(config))
        else:
            versions.append(ConfigurableVersion(config, digits))

    return versions


def build_versions(digits):
    return build_versions_from_configs(digits, VERSION_CONFIGS)


def evaluate_version(version_id, version, digits):
    per_digit = {
        digit: {"correct": 0, "total": 0}
        for digit in range(10)
    }
    start = time.perf_counter()
    correct = 0

    for image, target in zip(digits.images, digits.target):
        real_digit = int(target)
        predicted_digit = version.predict(image)

        per_digit[real_digit]["total"] += 1
        if predicted_digit == real_digit:
            correct += 1
            per_digit[real_digit]["correct"] += 1

    seconds = time.perf_counter() - start
    return VersionResult(version_id, version.name, correct, len(digits.images), seconds, per_digit)


def digit_accuracy(stats):
    if stats["total"] == 0:
        return 0
    return stats["correct"] / stats["total"]


def best_and_worst_digits(result):
    ranked = sorted(
        result.per_digit.items(),
        key=lambda item: digit_accuracy(item[1]),
    )
    return ranked[-1], ranked[0]


def format_top_items(outputs, limit=5):
    top_items = sorted(outputs.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{name}: {activation:.2f}" for name, activation in top_items[:limit])


def activation_bar(activation, width=12):
    filled = round(activation * width)
    return "#" * filled + "." * (width - filled)


def print_table(headers, rows):
    widths = [
        max(len(str(value)) for value in [header] + [row[index] for row in rows])
        for index, header in enumerate(headers)
    ]
    divider = "-+-".join("-" * width for width in widths)
    header_line = " | ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers))

    print(header_line)
    print(divider)
    for row in rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def print_results_table(results):
    rows = []
    for result in results:
        best, worst = best_and_worst_digits(result)
        best_digit, best_stats = best
        worst_digit, worst_stats = worst
        rows.append([
            result.version_id,
            result.name,
            f"{result.correct}/{result.total}",
            f"{result.accuracy:.2%}",
            f"{result.seconds:.4f}s",
            f"{best_digit} ({digit_accuracy(best_stats):.1%})",
            f"{worst_digit} ({digit_accuracy(worst_stats):.1%})",
        ])

    print_table(
        ["id", "version", "correct", "accuracy", "time", "best digit", "worst digit"],
        rows,
    )
    print()


def print_per_digit_table(result):
    rows = []
    for digit, stats in result.per_digit.items():
        rows.append([
            digit,
            f"{stats['correct']}/{stats['total']}",
            f"{digit_accuracy(stats):.2%}",
        ])

    print(f"Per-digit accuracy for: {result.name}")
    print_table(["digit", "correct", "accuracy"], rows)
    print()


def print_config_table(configs):
    rows = []
    for version_id, config in enumerate(configs):
        params = [
            f"{key}={value}"
            for key, value in config.items()
            if key not in {"name", "kind"}
        ]
        rows.append([
            version_id,
            config["name"],
            config["kind"],
            ", ".join(params) or "none",
        ])

    print_table(["id", "version", "kind", "parameters"], rows)
    print()


def editable_parameter_names(config):
    return [
        key
        for key, value in config.items()
        if key not in {"name", "kind"} and isinstance(value, (int, float))
    ]


def ask_number(prompt, current_value):
    while True:
        value = input(f"{prompt} [{current_value}]: ").strip()
        if value == "":
            return current_value

        try:
            if isinstance(current_value, int) and not isinstance(current_value, bool):
                return int(value)
            return float(value)
        except ValueError:
            print("Please type a valid number.")


def edit_version_config(configs):
    print("Current version parameters")
    print_config_table(configs)

    version_id = ask_int("Version id to edit", 0, len(configs) - 1, 1)
    config = configs[version_id]
    editable_names = editable_parameter_names(config)

    if not editable_names:
        print("That version has no editable numeric parameters.")
        print()
        return False

    print(f"Editable parameters for '{config['name']}':")
    for index, name in enumerate(editable_names):
        print(f"  {index}. {name} = {config[name]}")

    parameter_id = ask_int("Parameter id", 0, len(editable_names) - 1, 0)
    parameter_name = editable_names[parameter_id]
    old_value = config[parameter_name]
    new_value = ask_number(f"New value for {parameter_name}", old_value)
    config[parameter_name] = new_value

    print(f"Updated {config['name']}: {parameter_name} {old_value} -> {new_value}")
    print("Rebuilding versions and running the comparison again.")
    print()
    return True


def print_region(image, region):
    region = set(region)
    for row in range(len(image)):
        pixels = []
        for col in range(len(image[row])):
            if (row, col) in region:
                pixels.append(pixel_to_shade(image[row][col]) * 2)
            else:
                pixels.append("  ")
        print(f"      |{''.join(pixels)}|")


def print_top_regions(image, regions, outputs, label, limit=3):
    top_regions = sorted(outputs.items(), key=lambda item: item[1], reverse=True)
    print(f"  layer 1 detects {label}s:")
    for name, activation in top_regions[:limit]:
        print(f"    neuron '{name}' activation: {activation:.2f}")
        print_region(image, regions[name])


def print_output_matches(scores, layer_outputs, region_names, digit_layer, limit=3):
    active_regions = {
        name
        for name, activation in layer_outputs.items()
        if activation >= 0.65
    }

    print("  output layer strongest neurons:")
    for digit, score in scores[:limit]:
        neuron = digit_layer[digit]
        expected = [
            name
            for name, weight in zip(region_names, neuron.weights)
            if weight > 0
        ]
        detected = [name for name in expected if name in active_regions]
        missing = [name for name in expected if name not in active_regions]

        print(f"    digit {digit} score: {score:.3f}")
        print(f"      positive inputs: {', '.join(expected[:12])}")
        if len(expected) > 12:
            print(f"      ... plus {len(expected) - 12} more")
        print(f"      active positive inputs: {len(detected)}/{len(expected)}")
        if missing and len(missing) <= 12:
            print(f"      missing: {', '.join(missing)}")


def matching_indexes(digits, wanted_digit):
    return [
        index
        for index, target in enumerate(digits.target)
        if int(target) == wanted_digit
    ]


def analyze_sample(digits, version, index):
    if index < 0 or index >= len(digits.images):
        print(f"Index must be between 0 and {len(digits.images) - 1}.")
        return

    image = digits.images[index]
    real_digit = int(digits.target[index])
    details = version.predict_details(image)

    print()
    print(f"Version: {version.name}")
    print(f"Sample #{index} | real digit: {real_digit}")
    print("  original image:")
    print_digit(image)

    if version.upscale_factor > 1:
        print(f"  upscaled image used by this version ({version.image_size}x{version.image_size}):")
        print_digit(details.shown_image)

    version.print_layers(details)
    print()


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


def run_inspection_menu(digits, versions, configs):
    while True:
        print("Inspection menu")
        print("  1. Analyze exact dataset index")
        print("  2. Analyze examples by digit label")
        print("  3. Edit version parameters and compare again")
        print("  4. Show comparison table again")
        print("  5. Exit")

        choice = ask_int("Choose an option", 1, 5)
        print()

        if choice == 5:
            print("Bye.")
            return

        if choice == 4:
            return "show_table"

        if choice == 3:
            changed = edit_version_config(configs)
            return "rebuild" if changed else "show_table"

        version_id = ask_int("Version id", 0, len(versions) - 1, 0)
        version = versions[version_id]

        if choice == 1:
            index = ask_int("Dataset index", 0, len(digits.images) - 1, 0)
            analyze_sample(digits, version, index)
        elif choice == 2:
            digit = ask_int("Digit label", 0, 9, 5)
            limit = ask_int("How many examples", 1, 20, 3)
            for index in matching_indexes(digits, digit)[:limit]:
                analyze_sample(digits, version, index)


def main():
    parser = argparse.ArgumentParser(
        description="Compare configurable rudimentary digit-detector versions."
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="print the per-digit table for each version",
    )
    parser.add_argument(
        "--no-menu",
        action="store_true",
        help="only print comparison tables, then exit",
    )
    parser.add_argument(
        "--menu",
        action="store_true",
        help="force the inspection menu even when stdin is not interactive",
    )
    args = parser.parse_args()

    digits = load_digits_dataset()
    configs = [config.copy() for config in VERSION_CONFIGS]
    versions = build_versions_from_configs(digits, configs)

    print(f"Evaluating {len(versions)} versions on all {len(digits.images)} samples.")
    print("Edit VERSION_CONFIGS at the top of this file to add versions or tune parameters.")
    print()

    results = [
        evaluate_version(version_id, version, digits)
        for version_id, version in enumerate(versions)
    ]

    while True:
        print_results_table(results)

        if args.details:
            for result in results:
                print_per_digit_table(result)

        winner = max(results, key=lambda result: result.accuracy)
        fastest = min(results, key=lambda result: result.seconds)
        print("Summary")
        print(f"  best accuracy: {winner.name} ({winner.accuracy:.2%})")
        print(f"  fastest: {fastest.name} ({fastest.seconds:.4f} seconds)")
        print()

        should_open_menu = args.menu or (not args.no_menu and sys.stdin.isatty())
        if not should_open_menu:
            return

        action = run_inspection_menu(digits, versions, configs)
        if action == "rebuild":
            versions = build_versions_from_configs(digits, configs)
            results = [
                evaluate_version(version_id, version, digits)
                for version_id, version in enumerate(versions)
            ]
            continue

        if action != "show_table":
            return


if __name__ == "__main__":
    main()
