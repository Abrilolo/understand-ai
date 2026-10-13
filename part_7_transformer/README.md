# Part VII — Explicit causal attention

Run `python -m part_7_transformer.transformer --steps 600` from the
repository root. The 15,742-parameter decoder includes token and position
embeddings, four attention heads, residual connections, layer normalization,
a feed-forward block and output projection. PyTorch supplies tensor operations
and gradients; attention is explicit, without `nn.Transformer` or pretrained weights.

Queries represent what positions seek, keys represent matches and values carry
information to mix. We compute `Q Kᵀ / sqrt(head_width)`, mask future positions,
apply softmax and multiply by V. Tests verify that future changes leave earlier
logits unchanged, future attention is zero and attention rows sum to one.

The recorded run uses Part VI's lowercase corpus, tokenizer and split. AdamW trains
on random 32-character windows. Evaluation scores the same held-out target positions
once each. Validation NLL falls from 3.596 to **2.476**, versus **2.733** for Part VI
and **2.484** for its bigram baseline. One small run cannot establish superiority
over bigrams. Context, parameter count, objective, optimizer and budget differ,
so this comparison does not isolate attention's causal effect.

Samples contain more recognizable fragments, including “the flower,” but many
invented words remain. `results/metrics.json` contains unedited samples and metrics;
`results/model.pt` stores the trained state.

Reference: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
