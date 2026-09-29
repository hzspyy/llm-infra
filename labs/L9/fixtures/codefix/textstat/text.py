"""Small text helpers used by the code-fix agent tasks (L9.1)."""

import re


def tokenize(text):
    """Lowercase word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def word_freq(text, k):
    """Return the k most frequent words as (word, count), ties broken by word order."""
    counts = {}
    for w in tokenize(text):
        counts[w] = counts.get(w, 0) + 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return items[:k]


def truncate_middle(s, n):
    """Shorten s to at most n characters, keeping both ends and inserting '...'."""
    if n < 5:
        raise ValueError("n must be at least 5")
    if len(s) <= n:
        return s
    keep = n - 3
    left = (keep + 1) // 2
    right = keep - left
    return s[:left] + "..." + (s[len(s) - right:] if right else "")


def slugify(s):
    """Lowercase, hyphen-separated slug with no leading or trailing hyphen."""
    s = re.sub(r"[^a-z0-9]+", "-", s.lower())
    return s.strip("-")
