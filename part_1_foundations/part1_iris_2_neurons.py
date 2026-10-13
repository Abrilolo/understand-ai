"""
Part I - Iris experiment with only 2 hidden neurons.

This script intentionally uses a very small hidden layer:

    4 inputs -> 2 hidden neurons -> 3 output classes

The goal is to see how much the small network can learn and where it still
confuses Iris classes.
"""

from part1_iris_common import run_iris_experiment


def main():
    run_iris_experiment(
        hidden_neurons=2,
        epochs=2500,
        learning_rate=0.03,
        plot_filename="iris_2_hidden_neurons.png",
    )


if __name__ == "__main__":
    main()

