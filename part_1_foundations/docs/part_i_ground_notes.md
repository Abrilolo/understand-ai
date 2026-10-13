# Part I: The Ground - Knowledge Notes

These are the core answers we should be able to explain after finishing Part I.

## 1. What is a neuron? What is a weight? What is a bias? What is an activation function?

An artificial neuron is a small mathematical unit that receives input numbers, combines them, and produces one output number.

For inputs `x1, x2, ..., xn`, the neuron computes:

```text
z = x1*w1 + x2*w2 + ... + xn*wn + b
output = activation(z)
```

- A **weight** controls how strongly one input affects the neuron. A large positive weight makes that input push the output up. A large negative weight makes it push the output down.
- A **bias** is a constant added to the weighted sum. It shifts the decision boundary, similar to moving a line left, right, up, or down.
- An **activation function** transforms the raw weighted sum into the neuron's final output. Without activation functions, a network would only be a stack of linear equations. Common activations include step, sigmoid, tanh, and ReLU.

## 2. What is a neural network, and how is it more than the sum of its neurons?

A neural network is a group of neurons arranged in layers. Each layer receives values from the previous layer, transforms them, and passes the result forward.

A single neuron can only create a simple decision boundary. A network becomes more powerful because hidden layers combine many simple boundaries into more complex shapes. This lets the network approximate nonlinear patterns that one neuron cannot represent.

In other words, the power is not only in the neurons themselves, but in how their outputs become the inputs of other neurons.

## 3. What kinds of problems are neural networks good at, and what kinds are they bad at?

Neural networks are good at problems where patterns are hard to write as manual rules but can be learned from examples. Examples include image recognition, speech recognition, language modeling, recommendation systems, and classification or prediction from structured data.

They are weaker when:

- There is very little data.
- The data is noisy, biased, or not representative.
- The task requires exact symbolic reasoning or guaranteed correctness.
- The answer must be fully explainable step by step.
- The training and real-world data are very different.

Neural networks are pattern learners, not magic reasoning machines.

## 4. How do networks learn? What is backpropagation, in our own words?

A network learns by adjusting its weights and biases so its predictions become closer to the correct answers.

The basic loop is:

1. Make a prediction with the current weights.
2. Measure how wrong the prediction is using a loss function.
3. Work backward through the network to calculate how much each weight contributed to the error.
4. Change each weight a little in the direction that reduces the error.
5. Repeat many times.

Backpropagation is the "work backward" part. It applies the chain rule from calculus to compute gradients: numbers that tell us whether each weight should increase or decrease, and by how much.

## 5. What are common weight-adjustment methods, for example gradient descent and its variants?

The most basic method is **gradient descent**. It updates each parameter using:

```text
new_weight = old_weight - learning_rate * gradient
```

Common variants include:

- **Batch gradient descent**: uses the whole dataset before each update. Stable, but slow for large datasets.
- **Stochastic gradient descent (SGD)**: updates after one training example. Faster updates, but noisier.
- **Mini-batch gradient descent**: updates using small groups of examples. This is the most common practical approach.
- **Momentum**: remembers previous update directions so learning can move more smoothly.
- **RMSProp**: adapts learning rates based on recent gradient sizes.
- **Adam**: combines momentum-like behavior with adaptive learning rates. It is widely used because it works well in many situations.

## 6. What is the difference between AI, Machine Learning, Deep Learning, and an LLM?

**Artificial Intelligence (AI)** is the broad field of making computers perform tasks that seem intelligent, such as planning, recognizing patterns, making decisions, or understanding language.

**Machine Learning (ML)** is a subset of AI where systems learn behavior from data instead of being programmed with explicit rules for every case.

**Deep Learning (DL)** is a subset of machine learning that uses neural networks with multiple layers to learn complex representations.

An **LLM**, or Large Language Model, is a deep learning model trained on large amounts of text to predict and generate language. Modern LLMs usually use the transformer architecture and learn statistical patterns that let them summarize, translate, answer questions, write code, and use tools.

## 7. Why can a single neuron learn AND, but not XOR?

A single neuron with a threshold activation draws one straight decision boundary.

The AND gate is linearly separable. Only `(1, 1)` should output `1`, and the other three points should output `0`. We can draw one straight line that separates `(1, 1)` from `(0, 0)`, `(0, 1)`, and `(1, 0)`.

The XOR gate is not linearly separable. `(0, 1)` and `(1, 0)` should output `1`, while `(0, 0)` and `(1, 1)` should output `0`. These points sit diagonally from each other, so no single straight line can separate the `1` outputs from the `0` outputs.

To solve XOR, we need a small network with at least one hidden layer. The hidden neurons can create multiple boundaries, and the output neuron can combine them into a nonlinear decision region.

# Pseudocode for the First Task

This first coding idea focuses only on a single sigmoid neuron in pure Python, following the style of Michael Nielsen's explanation of perceptrons and sigmoid neurons.

## A. Single Neuron

The first step would be to represent a neuron as something that receives a few input numbers and gives back one output number.

Following Nielsen's explanation, the neuron should first combine the inputs with their weights. Each input has its own weight, so the neuron would multiply every input by its matching weight and add all those results together. After that, it would add a bias. The bias works like a knob that makes it easier or harder for the neuron to activate.

Once we have that final number, the neuron should pass it through an activation function. For a first version, the sigmoid function is a good choice because Nielsen uses it as a smoother version of a perceptron. Instead of jumping directly from `0` to `1`, the sigmoid gives a soft value between `0` and `1`.

So the rough idea is:

- Start with inputs, weights, and one bias.
- Multiply each input by its weight.
- Add those values together.
- Add the bias.
- Send the result through the sigmoid function.
- Return the final activation as the neuron's output.

If the output is close to `1`, the neuron is strongly activated. If it is close to `0`, the neuron is barely activated.
