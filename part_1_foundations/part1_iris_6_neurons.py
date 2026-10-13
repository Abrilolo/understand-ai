"""
Part I - Iris experiment with 6 hidden neurons.

This script uses a slightly larger hidden layer:

    4 inputs -> 6 hidden neurons -> 3 output classes

It is meant to compare against the 2-hidden-neuron version using the same
dataset split, training loop, and plotting code.
"""

from part1_iris_common import run_iris_experiment


def main():
    run_iris_experiment(
        hidden_neurons=6,
        epochs=2500,
        learning_rate=0.03,
        plot_filename="iris_6_hidden_neurons.png",
    )


if __name__ == "__main__":
    main()

