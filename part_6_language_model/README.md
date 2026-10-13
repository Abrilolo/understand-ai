# Part VI — A neural next-character predictor

Run `python -m part_6_language_model.language_model --epochs 15` from the
repository root. It imports Part I's exact `NeuralNetwork` and Part V's
`CharacterTokenizer`: four one-hot characters, 24 hidden sigmoid neurons and one
sigmoid score per character. The original squared-error backpropagation trains
the scores; normalization creates a sampling distribution afterward.

`corpus.txt` is original fictional educational text, approximately 1.8 KB, written
for this project. It is much smaller than the suggested corpus; the assignment
also permits our own text. Lowercasing precedes an ordered 80/20 split: 1,437 train
characters and 360 validation characters. The tokenizer sees training only.
Custom validation characters absent from training produce an explicit error.

The recorded run reduced training squared error from 1.064 to 0.804 and validation
NLL from 3.678 to **2.733 nats/token**. An add-one smoothed bigram scored **2.484**,
better than this network. Each validation target from position four is scored once.
`results/metrics.json` contains three actual samples, seeds, losses and corpus hash;
`results/model.json` stores weights and vocabulary. Samples show letter patterns
and spaces but are largely incoherent.

Unlike a count map, the network shares hidden features across contexts and learns
with gradients. The baseline stores transition counts conditioned on one previous
character. Both estimate statistical regularities; their representations differ.
This experiment does not show that neural models always outperform counting.
