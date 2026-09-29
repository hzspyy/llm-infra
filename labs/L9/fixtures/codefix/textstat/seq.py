"""Small sequence helpers used by the code-fix agent tasks (L9.1)."""


def chunk(xs, n):
    """Split xs into consecutive blocks of length n (last block may be shorter)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def cumsum(xs):
    """Return the inclusive prefix sums."""
    out = []
    acc = 0
    for x in xs:
        acc += x
        out.append(acc)
    return out


def dedupe_stable(xs):
    """Remove duplicates while keeping first-seen order."""
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def parse_ranges(s):
    """Expand '1-3,5' into [1, 2, 3, 5]."""
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def bucket(values, lo, hi, n):
    """Count values into n equal-width buckets covering [lo, hi]; values outside are ignored."""
    counts = [0] * n
    if hi <= lo:
        raise ValueError("hi must exceed lo")
    width = (hi - lo) / n
    for v in values:
        if v < lo or v > hi:
            continue
        idx = int((v - lo) / width)
        if idx >= n:
            idx = n - 1
        counts[idx] += 1
    return counts
