"""Next-character model reusing Part I's exact network and Part V's tokenizer.

Outputs are sigmoid scores trained with squared error (the Part I objective).
Normalize them only for sampling and evaluation; this is not softmax training.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from part_1_foundations.neural_network import NeuralNetwork
from part_5_tokenizer.tokenizer import CharacterTokenizer

CORPUS = Path(__file__).with_name('corpus.txt')


def load_data(path=CORPUS):
    text = Path(path).read_text(encoding='utf-8').lower()
    split = int(len(text) * 0.8)
    training, validation = text[:split], text[split:]
    tokenizer = CharacterTokenizer(training)
    # An explicit failure is preferable to silently discarding unknown text.
    return tokenizer, tokenizer.encode(training), tokenizer.encode(validation)


class NeuralLanguageModel:
    def __init__(self, vocabulary_size, context=4, hidden=24, seed=7):
        if min(vocabulary_size, context, hidden) < 1:
            raise ValueError('Model dimensions must be positive')
        random.seed(seed)
        self.vocabulary_size, self.context = vocabulary_size, context
        self.network = NeuralNetwork([context * vocabulary_size, hidden, vocabulary_size])

    def features(self, context):
        if len(context) != self.context:
            raise ValueError('Expected one complete context window')
        values = [0.0] * (self.context * self.vocabulary_size)
        for position, token in enumerate(context):
            values[position * self.vocabulary_size + token] = 1.0
        return values

    def probabilities(self, context):
        scores = self.network.forward(self.features(context))
        total = sum(scores)
        return [score / total for score in scores]

    def train(self, tokens, epochs=15, learning_rate=.15, seed=7):
        rng = random.Random(seed)
        indices = list(range(self.context, len(tokens)))
        if not indices or epochs < 1:
            raise ValueError('Need positive epochs and more tokens than context')
        history = []
        for epoch in range(epochs):
            rng.shuffle(indices)
            loss = 0.0
            for i in indices:
                scores = self.network.forward(self.features(tokens[i-self.context:i]))
                target = [float(j == tokens[i]) for j in range(self.vocabulary_size)]
                loss += sum((a-b)**2 for a, b in zip(scores, target))
                self.network.backward(target, learning_rate)
            history.append(loss / len(indices))
            print(f'epoch {epoch + 1}: squared error {history[-1]:.5f}', flush=True)
        return history

    def generate(self, prompt, count=300, seed=11):
        rng = random.Random(seed)
        tokens = list(prompt)
        if len(tokens) < self.context:
            raise ValueError('Prompt shorter than context')
        for _ in range(count):
            tokens.append(rng.choices(range(self.vocabulary_size), self.probabilities(tokens[-self.context:]))[0])
        return tokens


def mean_nll(probability_fn, tokens, context):
    return sum(-math.log(max(probability_fn(tokens[i-context:i])[tokens[i]], 1e-12))
               for i in range(context, len(tokens))) / (len(tokens) - context)


def run(epochs=15, corpus=CORPUS):
    tokenizer, train, validation = load_data(corpus)
    model = NeuralLanguageModel(len(tokenizer.characters))
    before = mean_nll(model.probabilities, validation, model.context)
    history = model.train(train, epochs)
    counts = defaultdict(Counter)
    for a, b in zip(train, train[1:]):
        counts[a][b] += 1

    def bigram(context):
        row = counts[context[-1]]
        total = sum(row.values()) + model.vocabulary_size
        return [(row[token] + 1) / total for token in range(model.vocabulary_size)]

    prompt = tokenizer.encode('the ')
    result = {'seed': 7, 'epochs': epochs, 'context': model.context,
              'corpus_sha256': hashlib.sha256(Path(corpus).read_bytes()).hexdigest(),
              'train_tokens': len(train), 'validation_tokens': len(validation),
              'preprocessing': 'lowercase; contiguous 80/20 character split',
              'objective': 'sum of squared errors on sigmoid scores', 'training_loss': history,
              'validation_nll_before': before,
              'validation_nll': mean_nll(model.probabilities, validation, model.context),
              'bigram_validation_nll': mean_nll(bigram, validation, model.context),
              'samples': [tokenizer.decode(model.generate(prompt, seed=seed)) for seed in (11, 12, 13)]}
    output = Path(__file__).parent / 'results'
    output.mkdir(exist_ok=True)
    checkpoint = {'characters': tokenizer.characters, 'context': model.context,
                  'layers': [[{'weights': neuron.weights, 'bias': neuron.bias}
                              for neuron in layer.neurons] for layer in model.network.layers]}
    (output / 'model.json').write_text(json.dumps(checkpoint), encoding='utf-8')
    (output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'training_loss'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--corpus', type=Path, default=CORPUS)
    args = parser.parse_args()
    run(args.epochs, args.corpus)
