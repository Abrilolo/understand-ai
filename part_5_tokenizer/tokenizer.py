"""Character tokenization and lossless UTF-8 byte-pair encoding, from scratch."""
from collections import Counter
import json
from pathlib import Path


class CharacterTokenizer:
    def __init__(self, text):
        self.characters = sorted(set(text))
        self.ids = {char: index for index, char in enumerate(self.characters)}

    def encode(self, text):
        return [self.ids[char] for char in text]

    def decode(self, tokens):
        return ''.join(self.characters[token] for token in tokens)


def merge_pair(tokens, pair, token_id):
    result = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            result.append(token_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


class BytePairTokenizer:
    """The first 256 tokens cover every byte, so unseen text has no unknowns."""
    def __init__(self):
        self.merges = []
        self.vocabulary = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size=300):
        if vocab_size < 256:
            raise ValueError('Byte vocabulary must contain at least 256 tokens')
        self.__init__()
        tokens = list(text.encode('utf-8'))
        while len(self.vocabulary) < vocab_size:
            counts = Counter(zip(tokens, tokens[1:]))
            if not counts:
                break
            pair, frequency = counts.most_common(1)[0]
            if frequency < 2:
                break
            token_id = len(self.vocabulary)
            self.merges.append(pair)
            self.vocabulary[token_id] = self.vocabulary[pair[0]] + self.vocabulary[pair[1]]
            tokens = merge_pair(tokens, pair, token_id)
        return self

    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        for token_id, pair in enumerate(self.merges, 256):
            tokens = merge_pair(tokens, pair, token_id)
        return tokens

    def decode(self, tokens):
        return b''.join(self.vocabulary[token] for token in tokens).decode('utf-8')

    def save(self, path):
        Path(path).write_text(json.dumps({'merges': self.merges}), encoding='utf-8')

    @classmethod
    def load(cls, path):
        tokenizer = cls()
        for pair in json.loads(Path(path).read_text(encoding='utf-8'))['merges']:
            pair = tuple(pair)
            tokenizer.merges.append(pair)
            tokenizer.vocabulary[len(tokenizer.vocabulary)] = (
                tokenizer.vocabulary[pair[0]] + tokenizer.vocabulary[pair[1]])
        return tokenizer


def main():
    examples = ['hello world, hello learner!', 'こんにちは、世界！', 'café 🐦 — new symbols']
    training = '\n'.join(examples[:2]) * 10
    character = CharacterTokenizer(training)
    bpe = BytePairTokenizer().train(training)
    results = []
    for text in examples:
        tokens = bpe.encode(text)
        assert bpe.decode(tokens) == text
        try:
            character_ids = character.encode(text)
            character_result = {'tokens': character_ids, 'decoded': character.decode(character_ids)}
        except KeyError as error:
            character_result = {'error': f'Unseen character: {error}'}
        results.append({'text': text, 'characters': character_result,
                        'utf8_bytes': len(text.encode('utf-8')), 'bpe_tokens': tokens,
                        'decoded': bpe.decode(tokens)})
    output = Path(__file__).parent / 'results'
    output.mkdir(exist_ok=True)
    bpe.save(output / 'tokenizer.json')
    (output / 'examples.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
