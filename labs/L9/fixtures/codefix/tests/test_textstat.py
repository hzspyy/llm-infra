import unittest

from textstat import (
    bucket,
    chunk,
    cumsum,
    dedupe_stable,
    median,
    normalize_scores,
    parse_ranges,
    percentile,
    rolling_max,
    running_mean,
    slugify,
    tokenize,
    truncate_middle,
    word_freq,
)


class TestStats(unittest.TestCase):
    def test_running_mean(self):
        self.assertEqual(running_mean([1, 2, 3]), [1.0, 1.5, 2.0])
        self.assertEqual(running_mean([2, 4, 6, 8]), [2.0, 3.0, 4.0, 5.0])
        self.assertEqual(running_mean([5]), [5.0])
        self.assertEqual(running_mean([-1, 1]), [-1.0, 0.0])
        self.assertEqual(running_mean([1, 2, 3, 4, 5]), [1.0, 1.5, 2.0, 2.5, 3.0])

    def test_median(self):
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([4, 1, 3, 2]), 2.5)
        self.assertEqual(median([7]), 7)
        self.assertEqual(median([-5, -1, -3]), -3)
        self.assertEqual(median([10, 2, 8, 4, 6, 12]), 7.0)

    def test_percentile(self):
        xs = [1, 2, 3, 4]
        self.assertEqual(percentile(xs, 0), 1)
        self.assertEqual(percentile(xs, 100), 4)
        self.assertEqual(percentile(xs, 50), 3)
        self.assertEqual(percentile(xs, 25), 2)
        self.assertEqual(percentile([9], 40), 9)
        with self.assertRaises(ValueError):
            percentile([], 50)

    def test_normalize_scores(self):
        self.assertEqual(normalize_scores({"a": 1, "b": 3}), {"a": 0.25, "b": 0.75})
        self.assertEqual(normalize_scores({"only": 2}), {"only": 1.0})
        self.assertEqual(normalize_scores({"a": 3, "b": 1, "c": 4}), {"a": 0.375, "b": 0.125, "c": 0.5})
        with self.assertRaises(ValueError):
            normalize_scores({"a": 0, "b": 0})
        with self.assertRaises(ValueError):
            normalize_scores({"a": -2, "b": 1})

    def test_rolling_max(self):
        self.assertEqual(rolling_max([1, 3, 2, 5], 2), [1, 3, 3, 5])
        self.assertEqual(rolling_max([4, 1, 2], 1), [4, 1, 2])
        self.assertEqual(rolling_max([2, 1, 3], 9), [2, 2, 3])
        self.assertEqual(rolling_max([5, 4, 3, 2], 3), [5, 5, 5, 4])
        with self.assertRaises(ValueError):
            rolling_max([1], 0)


class TestSeq(unittest.TestCase):
    def test_chunk(self):
        self.assertEqual(chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])
        self.assertEqual(chunk([1, 2, 3], 3), [[1, 2, 3]])
        self.assertEqual(chunk([1, 2], 1), [[1], [2]])
        self.assertEqual(chunk([], 4), [])
        with self.assertRaises(ValueError):
            chunk([1], 0)

    def test_cumsum(self):
        self.assertEqual(cumsum([1, 2, 3]), [1, 3, 6])
        self.assertEqual(cumsum([]), [])
        self.assertEqual(cumsum([-1, -2, 5]), [-1, -3, 2])
        self.assertEqual(cumsum([0, 0, 7]), [0, 0, 7])

    def test_dedupe_stable(self):
        self.assertEqual(dedupe_stable([1, 2, 1, 3, 2]), [1, 2, 3])
        self.assertEqual(dedupe_stable([]), [])
        self.assertEqual(dedupe_stable(["b", "a", "b"]), ["b", "a"])

    def test_parse_ranges(self):
        self.assertEqual(parse_ranges("1-3,5"), [1, 2, 3, 5])
        self.assertEqual(parse_ranges(" 7 , 9-10 "), [7, 9, 10])
        self.assertEqual(parse_ranges(""), [])
        self.assertEqual(parse_ranges("4"), [4])

    def test_bucket(self):
        self.assertEqual(bucket([0, 1, 2, 3, 4], 0, 4, 4), [1, 1, 1, 2])
        self.assertEqual(bucket([-1, 5, 2], 0, 4, 2), [0, 1])
        self.assertEqual(bucket([], 0, 1, 3), [0, 0, 0])
        with self.assertRaises(ValueError):
            bucket([1], 4, 4, 2)


class TestText(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("Hello, World! 42"), ["hello", "world", "42"])
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("a-b_c"), ["a", "b", "c"])

    def test_word_freq(self):
        self.assertEqual(word_freq("a b a c a b", 2), [("a", 3), ("b", 2)])
        self.assertEqual(word_freq("b a", 2), [("a", 1), ("b", 1)])
        self.assertEqual(word_freq("x y", 5), [("x", 1), ("y", 1)])
        self.assertEqual(word_freq("", 3), [])

    def test_truncate_middle(self):
        self.assertEqual(truncate_middle("abcdefghij", 7), "ab...ij")
        self.assertEqual(truncate_middle("abc", 5), "abc")
        self.assertEqual(truncate_middle("abcdef", 6), "abcdef")
        with self.assertRaises(ValueError):
            truncate_middle("abcdef", 4)

    def test_slugify(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")
        self.assertEqual(slugify("  A  B  "), "a-b")
        self.assertEqual(slugify("already-slug"), "already-slug")
        self.assertEqual(slugify("!!!"), "")


if __name__ == "__main__":
    unittest.main()
