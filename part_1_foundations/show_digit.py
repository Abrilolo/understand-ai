import argparse
import math
from pathlib import Path

from sklearn.datasets import load_digits


def parse_digit(value):
    try:
        digit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("digit must be a number from 0 to 9") from exc

    if digit < 0 or digit > 9:
        raise argparse.ArgumentTypeError("digit must be a number from 0 to 9")

    return digit


def ask_for_digit():
    while True:
        value = input("Which digit do you want to see? 0-9: ")
        try:
            return parse_digit(value)
        except argparse.ArgumentTypeError as error:
            print(error)


def load_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "This script needs matplotlib. Install it with:\n"
            "  .venv\\Scripts\\python.exe -m pip install matplotlib"
        ) from exc

    return plt


def matching_images(digits, wanted_digit):
    return [
        image
        for image, target in zip(digits.images, digits.target)
        if int(target) == wanted_digit
    ]


def save_images_grid(images, wanted_digit, columns, output_path):
    plt = load_matplotlib()

    rows = math.ceil(len(images) / columns)
    figure_width = columns * 1.2
    figure_height = rows * 1.2
    figure, axes = plt.subplots(rows, columns, figsize=(figure_width, figure_height))
    axes = axes.ravel()

    for index, axis in enumerate(axes):
        axis.axis("off")
        if index < len(images):
            axis.imshow(images[index], cmap="gray_r", interpolation="nearest")
            axis.set_title(str(index), fontsize=6)

    figure.suptitle(f"All images labeled as digit {wanted_digit} ({len(images)} total)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Show all sklearn digits images that match one label."
    )
    parser.add_argument(
        "digit",
        nargs="?",
        type=parse_digit,
        help="digit label to display, from 0 to 9",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=15,
        help="number of columns in the image grid",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="where to save the grid image",
    )
    args = parser.parse_args()

    wanted_digit = args.digit if args.digit is not None else ask_for_digit()
    digits = load_digits()
    images = matching_images(digits, wanted_digit)
    output_path = args.output or (
        Path(__file__).parent / "assets" / "digits" / f"digit_{wanted_digit}_examples.png"
    )

    print(f"Dataset shape: {digits.data.shape}")
    print(f"Found {len(images)} images labeled as digit {wanted_digit}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_images_grid(images, wanted_digit, args.columns, output_path)
    print(f"Saved image grid to: {output_path}")


if __name__ == "__main__":
    main()
