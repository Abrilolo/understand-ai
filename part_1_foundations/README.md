# Part I — The Ground

Neural networks from scratch: a neuron, AND/XOR gates, small networks, Iris and
additional handwritten-digit experiments. Neural-network math is plain Python.

## Structure

```text
part_1_foundations/
  neural_network.py             # reusable Neuron/NeuralNetwork + hierarchical digit detector
  single_neuron_digit_detector.py  # alternative, self-contained digit experiment
  print_region_layouts.py       # ASCII debug viewer for the stroke/segment regions
  compare_versions.py           # configuration comparison across detector versions
  show_digit.py                 # digit image grids
  show.py                       # simple viewer
  part1_and_neuron.py           # AND neuron and boundary plot
  xor_sneuron_and.py            # single-neuron XOR attempt
  part1_xor_network.py          # fixed and trained XOR networks
  part1_iris_2_neurons.py       # Iris with two hidden neurons
  part1_iris_6_neurons.py       # Iris with six hidden neurons
  part1_iris_common.py          # shared Iris implementation
  docs/                         # concept notes
  presentation/                 # original AI-history presentation
  assets/                       # generated plots
```

## Commands from the repository root

```powershell
.venv\Scripts\python.exe part_1_foundations/neural_network.py
.venv\Scripts\python.exe part_1_foundations/compare_versions.py --no-menu
.venv\Scripts\python.exe part_1_foundations/show_digit.py 5
.venv\Scripts\python.exe part_1_foundations/part1_and_neuron.py
.venv\Scripts\python.exe part_1_foundations/part1_xor_network.py
.venv\Scripts\python.exe part_1_foundations/part1_iris_2_neurons.py
.venv\Scripts\python.exe part_1_foundations/part1_iris_6_neurons.py
```

The [concept notes](docs/part_i_ground_notes.md) explain neurons, activations,
backpropagation, optimizers and AI/ML/DL/LLMs. The original
[AI-history slides](presentation/AI_History_From_Rules_to_Learning.html) are preserved.
The original Spanish guide remains as working notes.

AND admits a single line separating `(1,1)` from the other points. XOR places
positive labels on opposite corners, so one line cannot separate them. A 2–2–1
network combines hidden boundaries; the fixed network implements OR and NAND
followed by AND. The regenerated trained XOR outputs classify all four inputs
correctly. See `assets/plots/and_neuron_boundary.png`, `xor_neuron_boundary.png`
and `xor_trained_network.png`.

The verified 4–6–3 Iris run used 120 training and 30 test rows, training-only
normalization, 2,500 epochs and learning rate 0.03. It scored **96.67%** test
accuracy (29/30); one versicolor was classified as virginica. Scikit-learn loads
the public dataset; it does not train the model. This one deterministic split
is educational evidence, not a robust estimate across many possible splits.
