"""Benchmark exactly the Part I forward pass using identical float64 weights."""
import argparse
import json
import platform
import random
import statistics
from pathlib import Path
from time import perf_counter

from part_1_foundations.neural_network import NeuralNetwork


def benchmark(batch_size=256, repeats=15):
    import torch
    torch.set_num_threads(1)
    random.seed(7)
    network = NeuralNetwork([32, 64, 16])
    rows = [[random.uniform(-1, 1) for _ in range(32)] for _ in range(batch_size)]
    tensors = [(torch.tensor([n.weights for n in layer.neurons], dtype=torch.float64),
                torch.tensor([n.bias for n in layer.neurons], dtype=torch.float64))
               for layer in network.layers]
    inputs = torch.tensor(rows, dtype=torch.float64)

    def plain():
        return [network.forward(row) for row in rows]

    def tensor():
        values = inputs
        for weights, biases in tensors:
            values = torch.sigmoid(values @ weights.T + biases)
        return values

    with torch.inference_mode():
        expected = torch.tensor(plain(), dtype=torch.float64)
        error = float((expected - tensor()).abs().max())
        assert error < 1e-12, error
        for _ in range(5):
            plain()
            tensor()
        timings = {'python': [], 'torch': []}
        for i in range(repeats):
            # Alternate ordering to reduce systematic first/second bias.
            for name, operation in ([('python', plain), ('torch', tensor)] if i % 2
                                    else [('torch', tensor), ('python', plain)]):
                start = perf_counter()
                operation()
                timings[name].append(perf_counter() - start)
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {'shape': [32, 64, 16], 'batch_size': batch_size, 'repeats': repeats,
            'dtype': 'float64', 'device': 'cpu', 'torch_threads': 1,
            'python_version': platform.python_version(), 'torch_version': torch.__version__,
            'platform': platform.platform(), 'max_absolute_error': error,
            'seconds': timings, 'median_seconds': medians,
            'speedup': medians['python'] / medians['torch'],
            'scope': 'Forward only; tensor construction excluded; five warmups.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=15)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    result = [benchmark(size, args.repeats) for size in (1, 256)]
    output = Path(__file__).parent / 'results'
    output.mkdir(exist_ok=True)
    (output / 'benchmark.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
