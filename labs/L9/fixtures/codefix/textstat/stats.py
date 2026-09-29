"""Small numeric helpers used by the code-fix agent tasks (L9.1)."""


def running_mean(xs):
    """Return the running mean after each element."""
    acc = 0.0
    out = []
    for i, x in enumerate(xs):
        acc += x
        out.append(acc / (i + 1))
    return out


def median(xs):
    """Return the median of a non-empty sequence."""
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def percentile(xs, p):
    """Nearest-rank percentile, p in [0, 100]."""
    s = sorted(xs)
    if not s:
        raise ValueError("empty")
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    rank = int(round(p / 100.0 * (len(s) - 1)))
    return s[rank]


def normalize_scores(d):
    """Return a dict of non-negative weights normalized to sum 1."""
    total = sum(d.values())
    if total <= 0:
        raise ValueError("non-positive total")
    return {k: v / total for k, v in d.items()}


def rolling_max(xs, w):
    """Return the max over each trailing window of width w."""
    if w <= 0:
        raise ValueError("window must be positive")
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(max(xs[lo:i + 1]))
    return out
