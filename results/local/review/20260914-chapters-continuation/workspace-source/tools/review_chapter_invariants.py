"""Reproduce small mathematical and schema counterexamples for chapter review.

This command does not execute a lab main(), initialize CUDA, or rewrite results.
The output directory must be new. Run with the project Python and TMPDIR on
/Volumes/data. The GPTQ control uses the same fixed symmetric quantizer as the
lab; it checks the compensation step, not an official end-to-end quantizer.
"""

import argparse
import hashlib
import importlib.util
import json
from html.parser import HTMLParser
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_lab(relative_path):
    path = ROOT / relative_path
    name = "review_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def obs_reference(weight, hessian):
    """Recompute the inverse of the remaining Hessian after fixing each column."""
    current = weight.clone()
    output = torch.empty_like(current)
    scale = weight.abs().amax(dim=1) / 7
    for column in range(weight.shape[1]):
        inverse = torch.linalg.inv(hessian[column:, column:])
        quantized = (current[:, column] / scale).round().clamp(-7, 7) * scale
        error = (current[:, column] - quantized) / inverse[0, 0]
        current[:, column:] -= error[:, None] * inverse[0, :][None, :]
        output[:, column] = quantized
    return output


def cholesky_reference(weight, hessian):
    """GPTQ's upper-Cholesky compensation, with the lab's quantizer held fixed."""
    upper = torch.linalg.cholesky(torch.linalg.inv(hessian), upper=True)
    current = weight.clone()
    output = torch.empty_like(current)
    scale = weight.abs().amax(dim=1) / 7
    for column in range(weight.shape[1]):
        quantized = (current[:, column] / scale).round().clamp(-7, 7) * scale
        error = (current[:, column] - quantized) / upper[column, column]
        current[:, column:] -= error[:, None] * upper[column, column:][None, :]
        output[:, column] = quantized
    return output


class ArticleFigures(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_article = False
        self.in_code = False
        self.code_text = []
        self.svg_count = 0
        self.svg_code_count = 0

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.in_article = True
        if not self.in_article:
            return
        if tag == "svg":
            self.svg_count += 1
        if tag == "code":
            self.in_code = True
            self.code_text = []

    def handle_endtag(self, tag):
        if tag == "code" and self.in_code:
            self.svg_code_count += "<svg" in "".join(self.code_text)
            self.in_code = False
        if tag == "article":
            self.in_article = False

    def handle_data(self, data):
        if self.in_article and self.in_code:
            self.code_text.append(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    quant = load_lab("labs/L4/quantize_reference.py")
    alloc = load_lab("labs/L2/alloc_trace.py")
    combo = load_lab("labs/L2/alloc_graph_combo.py")
    result = {"torch_version": torch.__version__, "device": "CPU",
              "dtype": "float64", "random_inputs": False,
              "purpose": "mathematics and API-schema counterexamples; no GPU measurements"}

    x = torch.tensor([[1., 1.]], dtype=torch.float64)
    delta = torch.tensor([[1., 1.]], dtype=torch.float64)
    h = x.T @ x
    actual = (x @ delta.T).square().sum()
    full = torch.trace(delta @ h @ delta.T)
    block_diagonal = sum(delta[0, i].square() * h[i, i] for i in range(2))
    assert actual.item() == full.item() == 4 and block_diagonal.item() == 2
    result["cross_group_terms"] = {"X": x.tolist(), "delta_W": delta.tolist(),
                                   "actual_squared_error": actual.item(),
                                   "full_gram_formula": full.item(),
                                   "chapter_group_sum": block_diagonal.item()}

    weight = torch.tensor([[1., .31, .346]], dtype=torch.float64)
    hessian = torch.tensor([[4., 1., 1.], [1., 2., 1.], [1., 1., 2.]], dtype=torch.float64)
    observed = quant.gptq_quant(weight, hessian, bits=4, group_size=3, damp=0)[0]
    obs = obs_reference(weight, hessian)
    cholesky = cholesky_reference(weight, hessian)
    torch.testing.assert_close(obs, cholesky, rtol=0, atol=1e-12)
    result["gptq_compensation"] = {
        "W": weight.tolist(), "H": hessian.tolist(), "bits": 4, "group_size": 3,
        "damping": 0, "quantizer": "symmetric per-row absmax / 7, fixed across controls",
        "lab": observed.tolist(), "remaining_hessian_reference": obs.tolist(),
        "upper_cholesky_reference": cholesky.tolist(),
        "max_absolute_difference": (observed - obs).abs().max().item(),
        "reference_equivalence_tolerance": 1e-12,
        "official_compensation_source": "https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py#L99"}

    h2 = torch.tensor([[4., 1.], [1., 2.]], dtype=torch.float64)
    inverse = torch.linalg.inv(h2)
    error = .3 - 2 / 7
    result["two_column_compensation"] = {
        "H": h2.tolist(), "w0_minus_q0": error,
        "inverse_formula": float(-error * inverse[0, 1] / inverse[0, 0]),
        "correct_direct_hessian_formula": float(error * h2[0, 1] / h2[1, 1]),
        "chapter_direct_hessian_formula": float(-error * h2[0, 1] / h2[0, 0])}

    weight = torch.tensor([[1., 0., 6 / 7, 4 / 7]], dtype=torch.float64)
    group_cases = []
    for group in (4, 2):
        _, reconstructed, scale = quant.rtn_quant(weight, bits=4, group_size=group)
        group_cases.append({"group_size": group, "reconstructed": reconstructed.tolist(),
                            "scales": scale.tolist(), "weight_l2_error": (weight - reconstructed).norm().item()})
    result["smaller_group_not_monotonic"] = {"W": weight.tolist(), "cases": group_cases}

    weight = torch.tensor([[1., .2, .31, .7], [.4, .8, .1, .9]], dtype=torch.float64)
    x = torch.eye(4, dtype=torch.float64)
    rtn = quant.rtn_quant(weight, bits=4, group_size=2)[1]
    clipped = quant.clip_quant(weight, x, bits=4, group_size=2, ratios=[1.])[1]
    result["unclipped_control"] = {
        "W": weight.tolist(), "X": x.tolist(), "bits": 4, "group_size": 2,
        "ratios": [1.], "rtn": rtn.tolist(), "clipping": clipped.tolist(),
        "max_absolute_difference": (rtn - clipped).abs().max().item()}

    mib = 2**20
    snapshot = [{"total_size": 16 * mib, "blocks": [
        {"state": "active_allocated", "size": 4 * mib},
        {"state": "inactive", "size": 8 * mib},
        {"state": "active_awaiting_free", "size": 4 * mib}]}]
    with patch("torch.cuda.memory_snapshot", return_value=snapshot):
        alloc_observed = alloc.snapshot_summary()
        combo_observed = combo.snap()
    result["snapshot_schema"] = {
        "input_kind": "synthetic native-allocator segment using documented states",
        "input": snapshot, "alloc_trace_output": alloc_observed, "combo_output": combo_observed,
        "known_inactive_split_mib": 8, "known_pending_mib": 4,
        "actually_reusable_block_count": 1,
        "note": "inactive_split_bytes is a memory_stats counter, not a snapshot block state"}

    result["fp16_subnormals"] = {
        str(value): torch.tensor(value, dtype=torch.float16).item()
        for value in (1e-5, 1e-7, 1e-8)}
    initialized_before = torch.cuda.is_initialized()
    allocated = torch.cuda.memory_allocated()
    result["memory_allocated_without_cuda_init"] = {
        "initialized_before": initialized_before, "allocated": allocated,
        "initialized_after": torch.cuda.is_initialized(), "device": "CPU build; corroborated by pinned CUDA Python source"}

    pages = []
    for page in sorted((ROOT / "site").glob("L*/*.html")):
        reader = ArticleFigures()
        reader.feed(page.read_text())
        pages.append({"path": str(page.relative_to(ROOT)),
                      "sha256": hashlib.sha256(page.read_bytes()).hexdigest(),
                      "article_svg_count": reader.svg_count,
                      "article_svg_code_blocks": reader.svg_code_count})
    (args.output / "site_figures.json").write_text(json.dumps({
        "method": "HTMLParser on existing article DOM; no build or browser rendering",
        "pages": pages}, ensure_ascii=False, indent=2) + "\n")

    result["inputs"] = [{"path": relative,
                         "sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()}
                        for relative in ("labs/L4/quantize_reference.py", "labs/L2/alloc_trace.py",
                                         "labs/L2/alloc_graph_combo.py", "tools/review_chapter_invariants.py")]
    (args.output / "counterexamples.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
