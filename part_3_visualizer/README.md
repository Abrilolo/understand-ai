# Part III — A visible neural network

Run `python -m part_3_visualizer.visualize` from the repository root.
The demo imports the actual Part I `NeuralNetwork` and trains a 2–2–1 XOR network
with its original backpropagation. Press 1–4 to select an input, Space to pause,
and Escape to exit.

Nodes display numerical activations. Edges display weights; blue is positive and
coral is negative. Biases appear below neurons. Values change during training.
Moving dots illustrate forward flow, not measured execution latency. Combining
hidden neurons makes a nonlinear decision boundary possible. Training takes many
epochs; inspect the output to determine whether XOR has been learned.

The Flappy lab offers a second application. `results/preview.png` is a headless
rendering check of this standalone demo.
