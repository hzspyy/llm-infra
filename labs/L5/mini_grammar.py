#!/usr/bin/env python3
"""Finite-language constrained decoding, standard library only.

The language is {b'{"x":0}', b'{"x":1}'}. A token may span several bytes.
This enumerating matcher demonstrates the state contract, not a general JSON parser.
"""
import json
import math

VOCAB = [b'{', b'"x"', b':', b'0', b'1', b'}', b'{"x":', None, b'junk']
EOS = 7
LANGUAGE = (b'{"x":0}', b'{"x":1}')


class Matcher:
    def __init__(self, language=LANGUAGE):
        self.language = language
        self.prefix = b''
        self.done = False
        self.history = []

    def allowed(self, token_id):
        if self.done:
            return False
        piece = VOCAB[token_id]
        if piece is None:
            return self.prefix in self.language
        candidate = self.prefix + piece
        return any(word.startswith(candidate) for word in self.language)

    def mask(self):
        words = [0] * ((len(VOCAB) + 31) // 32)
        for token_id in range(len(VOCAB)):
            if self.allowed(token_id):
                words[token_id // 32] |= 1 << (token_id % 32)
        return words

    def accept(self, token_id):
        if not self.allowed(token_id):
            return False
        self.history.append((self.prefix, self.done))
        if token_id == EOS:
            self.done = True
        else:
            self.prefix += VOCAB[token_id]
        return True

    def rollback(self, count):
        if not 0 <= count <= len(self.history):
            raise ValueError('rollback exceeds accepted history')
        for _ in range(count):
            self.prefix, self.done = self.history.pop()


def mask_logits(logits, words):
    return [v if (words[i // 32] >> (i % 32)) & 1 else -math.inf
            for i, v in enumerate(logits)]


def self_check():
    # Exhaustively enumerate every tokenization admitted by this finite matcher.
    m = Matcher()
    completed = []
    def visit():
        if m.done:
            completed.append(m.prefix)
            return
        for i in range(len(VOCAB)):
            if m.allowed(i):
                old = (m.prefix, m.done, m.mask())
                assert m.accept(i)
                visit()
                m.rollback(1)
                assert old == (m.prefix, m.done, m.mask())
    visit()
    assert set(completed) == set(LANGUAGE)
    before = (m.prefix, m.mask())
    assert not m.accept(EOS) and not m.accept(8)
    assert before == (m.prefix, m.mask())
    try:
        m.rollback(1)
    except ValueError:
        pass
    else:
        raise AssertionError('rollback must reject invalid count')
    return len(completed)


if __name__ == '__main__':
    print('accepted tokenizations =', self_check())
    m = Matcher()
    logits = [1., 0., 0., 0., 1., 0., 2., 0., 10.]
    step = 0
    while not m.done:
        words = m.mask()
        masked = mask_logits(logits, words)
        selected = max(range(len(masked)), key=masked.__getitem__)
        print(json.dumps({'step': step, 'prefix': m.prefix.decode(),
              'words_hex': [f'{w:08x}' for w in words],
              'allowed_ids': [i for i in range(len(VOCAB)) if m.allowed(i)],
              'selected': selected, 'masked_logits': [str(v) for v in masked]}))
        assert m.accept(selected)
        step += 1
    print('output =', m.prefix.decode(), 'terminated =', m.done)
