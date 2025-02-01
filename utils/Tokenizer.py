import random as rnd


class Tokenizer:
    def __init__(self, sentences):
        self.sentences = sentences
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.build_vocab()

    def build_vocab(self):
        for sentence in self.sentences:
            for word in sentence.split():
                self.vocab.add(word)
        self.vocab = list(self.vocab)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, sentence):
        return [
            (
                self.word2idx[word]
                if word in self.word2idx
                else self._random_word_encoding()
            )
            for word in sentence.split()
        ]

    def decode(self, encoded):
        return [self.idx2word[idx] for idx in encoded]

    def _random_word_encoding(self):
        return rnd.choice(list(self.word2idx.values()))
