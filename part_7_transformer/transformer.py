"""A tiny decoder with explicit causal multi-head attention, not nn.Transformer."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from part_6_language_model.language_model import CORPUS, load_data


class CausalSelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError('Embedding width must be divisible by head count')
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.last_attention = None

    def forward(self, x):
        batch, length, width = x.shape
        # Each head independently compares its queries with keys.
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for t in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(width // self.heads)
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        attention = scores.masked_fill(future, float('-inf')).softmax(dim=-1)
        self.last_attention = attention.detach()
        mixed = (attention @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.projection(mixed)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, heads)
        self.feedforward = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.feedforward(self.norm2(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocabulary_size, context=32, width=32, heads=4):
        super().__init__()
        self.context = context
        self.tokens = nn.Embedding(vocabulary_size, width)
        self.positions = nn.Embedding(context, width)
        self.block = Block(width, heads)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocabulary_size)

    def forward(self, tokens):
        if tokens.shape[1] > self.context:
            raise ValueError('Input exceeds the context window')
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.tokens(tokens) + self.positions(positions)
        return self.output(self.norm(self.block(x)))

    @torch.no_grad()
    def generate(self, prompt, count=300, seed=11):
        self.eval()
        generator = torch.Generator().manual_seed(seed)
        tokens = torch.tensor([prompt], dtype=torch.long)
        for _ in range(count):
            probabilities = self(tokens[:, -self.context:])[:, -1].softmax(-1)
            token = torch.multinomial(probabilities, 1, generator=generator)
            tokens = torch.cat((tokens, token), dim=1)
        return tokens[0].tolist()


@torch.no_grad()
def evaluate(model, tokens, start=4):
    """Score each target once, matching Part VI's validation targets."""
    model.eval()
    losses = []
    for i in range(start, len(tokens)):
        x = torch.tensor([tokens[max(0, i-model.context):i]], dtype=torch.long)
        loss = F.cross_entropy(model(x)[:, -1], torch.tensor([tokens[i]]))
        losses.append(float(loss))
    return sum(losses) / len(losses)


def run(steps=600, corpus=CORPUS):
    if steps < 1:
        raise ValueError('steps must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(7)
    tokenizer, train, validation = load_data(corpus)
    model = TinyTransformer(len(tokenizer.characters))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    data = torch.tensor(train)
    before = evaluate(model, validation)
    history = []
    model.train()
    for step in range(steps):
        starts = torch.randint(len(data) - model.context, (16,))
        x = torch.stack([data[i:i+model.context] for i in starts])
        y = torch.stack([data[i+1:i+model.context+1] for i in starts])
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(tokenizer.characters)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == steps-1:
            history.append({'step': step + 1, 'training_nll': float(loss.detach())})
            print(history[-1], flush=True)
    after = evaluate(model, validation)
    samples = [tokenizer.decode(model.generate(tokenizer.encode('the '), seed=seed))
               for seed in (11, 12, 13)]
    result = {'seed': 7, 'steps': steps, 'context': model.context,
              'parameters': sum(p.numel() for p in model.parameters()),
              'corpus_sha256': hashlib.sha256(Path(corpus).read_bytes()).hexdigest(),
              'preprocessing': 'lowercase; contiguous 80/20 character split',
              'train_tokens': len(train), 'validation_tokens': len(validation),
              'validation_nll_before': before, 'validation_nll': after,
              'training_history': history, 'samples': samples}
    output = Path(__file__).parent / 'results'
    output.mkdir(exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'characters': tokenizer.characters,
                'context': model.context, 'width': 32, 'heads': 4}, output / 'model.pt')
    (output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--corpus', type=Path, default=CORPUS)
    args = parser.parse_args()
    run(args.steps, args.corpus)
