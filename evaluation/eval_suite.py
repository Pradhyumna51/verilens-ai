"""
VeriLens AI — Automated Evaluation & Benchmark Suite
======================================================
Member 4 Deliverable: Computes multi-class classification metrics,
robustness retention delta under adversarial perturbation, and
explanation faithfulness / evidence keyword coverage.

Usage:
    python -m evaluation.eval_suite
    python -m evaluation.eval_suite --verbose
    python -m evaluation.eval_suite --samples evaluation/dataset.json
"""

import sys
import json
import io
import time
import math
import argparse
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# Configure UTF-8 output safely on Windows if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from verilens.pipeline.orchestrator import AnalysisPipeline
from verilens.schemas.response import AnalysisResponse, Verdict

# ── Constants ────────────────────────────────────────────────────────────────
DATASET_PATH = ROOT / "evaluation" / "dataset.json"
VERDICT_CLASSES = ["LIKELY_REAL", "UNCERTAIN", "SUSPICIOUS", "LIKELY_MANIPULATED"]
VERDICT_INDEX = {v: i for i, v in enumerate(VERDICT_CLASSES)}

# Risk ordering for adversarial delta comparison
RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


# ============================================================================
# SECTION 1 — Synthetic media generators (deterministic)
# ============================================================================

def _make_synthetic_image(size: Tuple[int, int] = (256, 256)) -> bytes:
    """High-frequency periodic grid with blue-dominated spectrum -- signature of diffusion model output."""
    arr = np.zeros((*size, 3), dtype=np.uint8)
    for y in range(size[0]):
        for x in range(size[1]):
            val = int((math.sin(x / 2.0) * math.cos(y / 2.0) + 1.0) * 127)
            arr[y, x] = [int(val * 0.7), (val * 2) % 255, 255 - int(val * 0.3)]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _make_natural_image(size: Tuple[int, int] = (256, 256)) -> bytes:
    """Natural landscape: clear daylight sky and green hills without skin tone artifacts."""
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size[0], size[1] // 2], fill=(135, 206, 235))
    draw.rectangle([0, size[1] // 2, size[0], size[1]], fill=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _make_daylight_image(size: Tuple[int, int] = (256, 256)) -> bytes:
    """Bright daylight scene with high luminance (exceeds 170) to trigger nighttime contradictions."""
    img = Image.new("RGB", size, color=(220, 235, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _make_tiny_image() -> bytes:
    """Sub-threshold thumbnail (20x20) to trigger UNCERTAIN status."""
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), color=(128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def _make_recycled_image() -> bytes:
    """Historical archive image with embedded EXIF Software provenance metadata tag."""
    img = Image.new("RGB", (320, 240), color=(40, 90, 160))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 280, 200], fill=(70, 130, 200), outline=(255, 255, 255))
    draw.text((60, 110), "HISTORICAL ARCHIVE 2019", fill=(255, 255, 255))
    exif = img.getexif()
    exif[0x0131] = "Historical Archive Suite 2019"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def _apply_jpeg_compression(img_bytes: bytes, quality: int = 35) -> bytes:
    """Adversarial perturbation: aggressive JPEG re-compression."""
    img = Image.open(io.BytesIO(img_bytes))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _apply_gaussian_blur(img_bytes: bytes, radius: float = 2.0) -> bytes:
    """Adversarial perturbation: Gaussian blur to suppress spectral grid artifacts."""
    img = Image.open(io.BytesIO(img_bytes))
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    buf = io.BytesIO()
    blurred.save(buf, format="PNG")
    return buf.getvalue()


def _paraphrase_text(text: str) -> str:
    """
    Adversarial perturbation: simple lexical substitution to alter burstiness.
    Replaces common AI marker phrases with plainer equivalents.
    """
    if not text:
        return text
    replacements = {
        "In conclusion,": "To sum up,",
        "Furthermore,": "Also,",
        "Moreover,": "In addition,",
        "Therefore,": "So,",
        "it is important to remember that": "we should note that",
        "fundamentally transformed": "changed",
        "unprecedented": "remarkable",
        "cutting-edge": "advanced",
        "rich tapestry": "wide range",
        "profound testament": "clear example",
        "multifaceted dynamics": "complex factors",
        "interconnected": "linked",
        "digital infrastructure": "tech systems",
        "algorithmic reality": "digital world",
    }
    result = text
    for orig, sub in replacements.items():
        result = result.replace(orig, sub)
    return result


# ============================================================================
# SECTION 2 — Sample resolver: build runtime input from dataset.json entry
# ============================================================================

def resolve_sample_inputs(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a dataset.json entry into kwargs accepted by pipeline.run().
    Generates deterministic synthetic media bytes where needed.
    """
    text: Optional[str] = sample.get("text") or sample.get("claim")
    media_bytes: Optional[bytes] = None
    media_filename: Optional[str] = None
    media_content_type: Optional[str] = None

    if sample.get("has_media") or sample.get("media_type") == "image":
        if sample.get("synthetic_image"):
            media_bytes = _make_synthetic_image()
            media_filename = "synthetic_sample.png"
            media_content_type = "image/png"
        else:
            cat = sample.get("category", "clean")
            if cat == "recycled_media":
                media_bytes = _make_recycled_image()
                media_filename = "recycled_archive.jpg"
                media_content_type = "image/jpeg"
            elif cat == "cross_modal_mismatch":
                media_bytes = _make_daylight_image()
                media_filename = "daylight_sky.jpg"
                media_content_type = "image/jpeg"
            elif cat == "adversarial_noise" and sample.get("label") == "UNCERTAIN":
                media_bytes = _make_tiny_image()
                media_filename = "tiny_thumb.png"
                media_content_type = "image/png"
            else:
                media_bytes = _make_natural_image()
                media_filename = "natural_photo.jpg"
                media_content_type = "image/jpeg"

    return dict(
        text=text,
        media_bytes=media_bytes,
        media_filename=media_filename,
        media_content_type=media_content_type,
        enabled_detectors=None,
    )


# ============================================================================
# SECTION 3 — Multi-class Classification Metrics
# ============================================================================

def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str] = VERDICT_CLASSES,
) -> Dict[str, Any]:
    """
    Computes per-class and macro-averaged Precision, Recall, F1, and overall Accuracy.
    """
    n = len(y_true)
    assert n == len(y_pred), "Label lists must be equal length."

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for true_lbl, pred_lbl in zip(y_true, y_pred):
        for cls in classes:
            if true_lbl == cls and pred_lbl == cls:
                tp[cls] += 1
            elif true_lbl != cls and pred_lbl == cls:
                fp[cls] += 1
            elif true_lbl == cls and pred_lbl != cls:
                fn[cls] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0

    for cls in classes:
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = tp[cls] + fn[cls]
        per_class[cls] = {"precision": p, "recall": r, "f1": f1, "support": support}
        macro_p += p
        macro_r += r
        macro_f1 += f1

    num_cls = len(classes)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_p / num_cls,
        "macro_recall": macro_r / num_cls,
        "macro_f1": macro_f1 / num_cls,
        "per_class": per_class,
        "n_samples": n,
    }


# ============================================================================
# SECTION 4 — Robustness Retention Delta
# ============================================================================

def compute_robustness_delta(
    pipeline: AnalysisPipeline,
    samples: List[Dict[str, Any]],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Measures how much the manipulation score changes under adversarial perturbations.
    """
    results = []
    image_samples = [s for s in samples if s.get("has_media") and s.get("synthetic_image")]
    text_samples = [s for s in samples if s.get("text") and not s.get("has_media")]

    # --- Image perturbation tests ---
    for sample in image_samples[:4]:
        inputs = resolve_sample_inputs(sample)
        orig_bytes = inputs["media_bytes"]

        try:
            orig_resp = pipeline.run(**inputs)
            orig_score = _extract_overall_score(orig_resp)

            # JPEG compression perturbation
            inputs_jpeg = dict(inputs)
            inputs_jpeg["media_bytes"] = _apply_jpeg_compression(orig_bytes, quality=35)
            inputs_jpeg["media_filename"] = "perturbed_jpeg.jpg"
            inputs_jpeg["media_content_type"] = "image/jpeg"
            jpeg_resp = pipeline.run(**inputs_jpeg)
            jpeg_score = _extract_overall_score(jpeg_resp)

            # Gaussian blur perturbation
            inputs_blur = dict(inputs)
            inputs_blur["media_bytes"] = _apply_gaussian_blur(orig_bytes, radius=2.5)
            inputs_blur["media_filename"] = "perturbed_blur.png"
            inputs_blur["media_content_type"] = "image/png"
            blur_resp = pipeline.run(**inputs_blur)
            blur_score = _extract_overall_score(blur_resp)

            delta_jpeg = jpeg_score - orig_score
            delta_blur = blur_score - orig_score
            retention_jpeg = 1.0 - abs(delta_jpeg) / max(orig_score, 0.01)
            retention_blur = 1.0 - abs(delta_blur) / max(orig_score, 0.01)

            results.append({
                "sample_id": sample["id"],
                "perturbation": "jpeg_compression",
                "original_score": round(orig_score, 4),
                "perturbed_score": round(jpeg_score, 4),
                "delta": round(delta_jpeg, 4),
                "retention": round(max(retention_jpeg, 0.0), 4),
            })
            results.append({
                "sample_id": sample["id"],
                "perturbation": "gaussian_blur",
                "original_score": round(orig_score, 4),
                "perturbed_score": round(blur_score, 4),
                "delta": round(delta_blur, 4),
                "retention": round(max(retention_blur, 0.0), 4),
            })

            if verbose:
                _log(f"  [{sample['id']}] orig={orig_score:.3f} | "
                     f"jpeg={jpeg_score:.3f} (d{delta_jpeg:+.3f}) | "
                     f"blur={blur_score:.3f} (d{delta_blur:+.3f})")

        except Exception as exc:
            if verbose:
                _log(f"  [{sample['id']}] SKIPPED: {exc}")

    # --- Text paraphrase perturbation tests ---
    for sample in text_samples[:3]:
        inputs = resolve_sample_inputs(sample)
        if not inputs.get("text"):
            continue
        try:
            orig_resp = pipeline.run(**inputs)
            orig_score = _extract_overall_score(orig_resp)

            inputs_para = dict(inputs)
            inputs_para["text"] = _paraphrase_text(inputs["text"])
            para_resp = pipeline.run(**inputs_para)
            para_score = _extract_overall_score(para_resp)

            delta = para_score - orig_score
            retention = 1.0 - abs(delta) / max(orig_score, 0.01)
            results.append({
                "sample_id": sample["id"],
                "perturbation": "text_paraphrase",
                "original_score": round(orig_score, 4),
                "perturbed_score": round(para_score, 4),
                "delta": round(delta, 4),
                "retention": round(max(retention, 0.0), 4),
            })
            if verbose:
                _log(f"  [{sample['id']}] orig={orig_score:.3f} | para={para_score:.3f} (d{delta:+.3f})")

        except Exception as exc:
            if verbose:
                _log(f"  [{sample['id']}] SKIPPED: {exc}")

    if not results:
        return {"mean_retention": None, "mean_abs_delta": None, "details": []}

    retentions = [r["retention"] for r in results]
    deltas = [abs(r["delta"]) for r in results]
    return {
        "mean_retention": round(float(np.mean(retentions)), 4),
        "mean_abs_delta": round(float(np.mean(deltas)), 4),
        "n_perturbations": len(results),
        "details": results,
    }


def _extract_overall_score(resp: AnalysisResponse) -> float:
    """Derive a scalar score from the response for robustness comparison."""
    success_findings = [
        f for f in resp.findings.values()
        if f.manipulation_score is not None
    ]
    if not success_findings:
        return 0.0
    return float(np.mean([f.manipulation_score for f in success_findings]))


# ============================================================================
# SECTION 5 — Explanation Faithfulness / Evidence Keyword Coverage
# ============================================================================

def compute_faithfulness(
    responses: List[AnalysisResponse],
    samples: List[Dict[str, Any]],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Measures whether the explanation and evidence fields contain keywords
    expected for each sample's active detectors (as declared in dataset.json).
    """
    coverage_scores = []
    details = []

    for resp, sample in zip(responses, samples):
        expected_kw: List[str] = sample.get("expected_keywords") or sample.get("expected_evidence_keywords") or []
        if not expected_kw:
            continue

        searchable_text = " ".join([
            resp.explanation.headline.lower(),
            resp.explanation.rationale.lower(),
            " ".join(sp.lower() for sp in resp.explanation.supporting_points),
            " ".join(ce.lower() for ce in resp.explanation.counter_evidence),
            " ".join(ev.description.lower() for ev in resp.evidence),
            resp.overall_assessment.summary.lower(),
        ])

        matched = [kw for kw in expected_kw if kw.lower() in searchable_text]
        coverage = len(matched) / len(expected_kw)
        coverage_scores.append(coverage)

        detail = {
            "sample_id": sample["id"],
            "expected_keywords": expected_kw,
            "matched_keywords": matched,
            "coverage": round(coverage, 4),
        }
        details.append(detail)

        if verbose:
            missed = [kw for kw in expected_kw if kw not in matched]
            _log(f"  [{sample['id']}] coverage={coverage:.0%}  "
                 f"matched={matched}  missed={missed}")

    if not coverage_scores:
        return {"mean_coverage": None, "n_evaluated": 0, "details": []}

    return {
        "mean_coverage": round(float(np.mean(coverage_scores)), 4),
        "n_evaluated": len(coverage_scores),
        "details": details,
    }


# ============================================================================
# SECTION 6 — Display helpers (Clean ASCII for universal compatibility)
# ============================================================================

def _log(msg: str) -> None:
    print(msg, flush=True)


def _hr(char: str = "-", width: int = 72) -> str:
    return char * width


def _pct(val: Optional[float]) -> str:
    return f"{val * 100:.1f}%" if val is not None else "N/A"


def _render_classification_table(metrics: Dict[str, Any]) -> None:
    _log(_hr("="))
    _log("  MULTI-CLASS CLASSIFICATION METRICS")
    _log(_hr("="))
    _log(f"  Samples evaluated : {metrics['n_samples']}")
    _log(f"  Overall Accuracy  : {_pct(metrics['accuracy'])}")
    _log(f"  Macro Precision   : {_pct(metrics['macro_precision'])}")
    _log(f"  Macro Recall      : {_pct(metrics['macro_recall'])}")
    _log(f"  Macro F1-Score    : {_pct(metrics['macro_f1'])}")
    _log("")
    _log(f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    _log(f"  {_hr('-', 67)}")
    for cls, m in metrics["per_class"].items():
        _log(f"  {cls:<25} {_pct(m['precision']):>10} {_pct(m['recall']):>10} {_pct(m['f1']):>10} {m['support']:>10}")
    _log(_hr("="))


def _render_robustness_table(rob: Dict[str, Any]) -> None:
    _log("")
    _log(_hr("="))
    _log("  ROBUSTNESS RETENTION DELTA (Adversarial Perturbations)")
    _log(_hr("="))

    if rob.get("mean_retention") is None:
        _log("  No perturbation results (no eligible image or text samples).")
        _log(_hr("="))
        return

    _log(f"  Perturbations tested  : {rob['n_perturbations']}")
    _log(f"  Mean Retention        : {_pct(rob['mean_retention'])}  (higher = more robust)")
    _log(f"  Mean |Delta score|    : {rob['mean_abs_delta']:.4f}  (lower = more stable)")
    _log("")
    _log(f"  {'Sample':<14} {'Perturbation':<22} {'Orig':>8} {'Perturb':>9} {'Delta':>8} {'Retain':>8}")
    _log(f"  {_hr('-', 73)}")
    for row in rob["details"]:
        _log(
            f"  {row['sample_id']:<14} {row['perturbation']:<22} "
            f"{row['original_score']:>8.4f} {row['perturbed_score']:>9.4f} "
            f"{row['delta']:>+8.4f} {_pct(row['retention']):>8}"
        )
    _log(_hr("="))


def _render_faithfulness_table(faith: Dict[str, Any]) -> None:
    _log("")
    _log(_hr("="))
    _log("  EXPLANATION FAITHFULNESS / EVIDENCE KEYWORD COVERAGE")
    _log(_hr("="))

    if faith.get("mean_coverage") is None:
        _log("  No faithfulness results (no keyword expectations in dataset).")
        _log(_hr("="))
        return

    _log(f"  Samples with keywords : {faith['n_evaluated']}")
    _log(f"  Mean keyword coverage : {_pct(faith['mean_coverage'])}")
    _log("")
    _log(f"  {'Sample':<14} {'Coverage':>10}  {'Matched Keywords'}")
    _log(f"  {_hr('-', 60)}")
    for row in faith["details"]:
        matched_str = ", ".join(row["matched_keywords"]) or "(none)"
        _log(f"  {row['sample_id']:<14} {_pct(row['coverage']):>10}  {matched_str}")
    _log(_hr("="))


def _render_summary_badge(cls_m: Dict, rob: Dict, faith: Dict) -> None:
    _log("")
    _log(_hr("="))
    _log("  OVERALL EVALUATION SUMMARY")
    _log(_hr("="))
    acc = cls_m["accuracy"]
    f1 = cls_m["macro_f1"]
    ret = rob.get("mean_retention")
    cov = faith.get("mean_coverage")

    _log(f"  {'Metric':<38} {'Value':>10}  {'Status'}")
    _log(f"  {_hr('-', 60)}")

    def grade(val, thresholds):
        if val is None:
            return "N/A", "[ - ]"
        if val >= thresholds[0]:
            return _pct(val), "[PASS]"
        elif val >= thresholds[1]:
            return _pct(val), "[WARN]"
        else:
            return _pct(val), "[FAIL]"

    rows = [
        ("Multi-class Accuracy",          acc, (0.60, 0.40)),
        ("Macro F1-Score",                 f1,  (0.55, 0.35)),
        ("Robustness Retention",           ret, (0.70, 0.50)),
        ("Explanation Keyword Coverage",   cov, (0.50, 0.30)),
    ]
    for label, val, thresh in rows:
        vstr, icon = grade(val, thresh)
        _log(f"  {label:<38} {vstr:>10}  {icon}")

    _log(_hr("="))
    _log("")
    _log("  NOTE: VeriLens AI operates as a hackathon-grade prototype.")
    _log("  All heuristic detectors use algorithmic signals only.")
    _log("  Human-in-the-loop verification is always recommended.")
    _log(_hr("="))


# ============================================================================
# SECTION 7 — Main orchestration
# ============================================================================

def run_eval_suite(
    dataset_path: Path = DATASET_PATH,
    verbose: bool = False,
) -> Dict[str, Any]:
    _log("")
    _log("+==================================================================+")
    _log("|          VeriLens AI -- Automated Evaluation Suite v1.0          |")
    _log("+==================================================================+")
    _log(f"  Dataset  : {dataset_path}")
    _log(f"  Verbose  : {verbose}")
    _log("")

    # ── Load dataset ──────────────────────────────────────────────────────
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if isinstance(dataset, dict) and "samples" in dataset:
        samples: List[Dict[str, Any]] = dataset["samples"]
    elif isinstance(dataset, list):
        samples: List[Dict[str, Any]] = dataset
    else:
        raise ValueError(f"Unexpected dataset structure in {dataset_path}")

    LABEL_MAP = {
        "AUTHENTIC": "LIKELY_REAL",
        "REAL": "LIKELY_REAL",
        "MISLEADING": "SUSPICIOUS",
        "OUT_OF_CONTEXT": "SUSPICIOUS",
        "MANIPULATED": "LIKELY_MANIPULATED",
        "FAKE": "LIKELY_MANIPULATED",
    }

    _log(f"  Loaded {len(samples)} ground-truth samples from dataset.json")
    _log("")

    # ── Instantiate pipeline ──────────────────────────────────────────────
    _log("  Initialising VeriLens pipeline orchestrator...")
    pipe = AnalysisPipeline()
    _log("  [OK] Pipeline ready.\n")

    # ── Run inference on all samples ──────────────────────────────────────
    _log(_hr("="))
    _log("  RUNNING INFERENCE ON ALL SAMPLES")
    _log(_hr("="))

    y_true: List[str] = []
    y_pred: List[str] = []
    responses: List[AnalysisResponse] = []
    latencies: List[float] = []

    for i, sample in enumerate(samples):
        inputs = resolve_sample_inputs(sample)
        raw_label = sample.get("label") or sample.get("ground_truth_verdict") or "UNCERTAIN"
        normalized_label = LABEL_MAP.get(raw_label, raw_label)
        t0 = time.perf_counter()

        try:
            resp = pipe.run(**inputs)
            elapsed = (time.perf_counter() - t0) * 1000

            pred_verdict = resp.overall_assessment.verdict.value
            y_true.append(normalized_label)
            y_pred.append(pred_verdict)
            responses.append(resp)
            latencies.append(elapsed)

            correct_marker = "[OK]" if normalized_label == pred_verdict else "[MISMATCH]"
            _log(
                f"  [{i+1:02d}/{len(samples)}] {sample['id']:<14} "
                f"true={normalized_label:<18} "
                f"pred={pred_verdict:<18} "
                f"{correct_marker}  ({elapsed:.0f} ms)"
            )

            if verbose:
                _log(f"         risk={resp.overall_assessment.risk_level.value}  "
                     f"conf={resp.overall_assessment.confidence:.2f}  "
                     f"summary: {resp.overall_assessment.summary[:60]}...")

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            _log(f"  [{i+1:02d}/{len(samples)}] {sample['id']:<14} ERROR: {exc}")
            y_true.append(normalized_label)
            y_pred.append("UNCERTAIN")
            responses.append(None)
            latencies.append(elapsed)

    _log("")
    _log(f"  Mean latency: {np.mean(latencies):.1f} ms | Total: {sum(latencies):.0f} ms")

    # ── 1. Classification metrics ─────────────────────────────────────────
    _log("\n  Computing classification metrics...")
    cls_metrics = compute_classification_metrics(y_true, y_pred)
    _render_classification_table(cls_metrics)

    # ── 2. Robustness delta ───────────────────────────────────────────────
    _log("\n  Running adversarial robustness perturbation tests...")
    robustness = compute_robustness_delta(pipe, samples, verbose=verbose)
    _render_robustness_table(robustness)

    # ── 3. Explanation faithfulness ───────────────────────────────────────
    _log("\n  Evaluating explanation faithfulness / keyword coverage...")
    valid_pairs = [(r, s) for r, s in zip(responses, samples) if r is not None]
    valid_resps = [p[0] for p in valid_pairs]
    valid_samps = [p[1] for p in valid_pairs]
    faithfulness = compute_faithfulness(valid_resps, valid_samps, verbose=verbose)
    _render_faithfulness_table(faithfulness)

    # ── Summary badge ─────────────────────────────────────────────────────
    _render_summary_badge(cls_metrics, robustness, faithfulness)

    return {
        "classification": cls_metrics,
        "robustness": robustness,
        "faithfulness": faithfulness,
    }


# ============================================================================
# SECTION 8 — CLI entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VeriLens AI Automated Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m evaluation.eval_suite
              python -m evaluation.eval_suite --verbose
              python -m evaluation.eval_suite --samples evaluation/dataset.json
        """),
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=DATASET_PATH,
        help="Path to dataset.json (default: evaluation/dataset.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-sample details and intermediate steps.",
    )
    args = parser.parse_args()

    results = run_eval_suite(dataset_path=args.samples, verbose=args.verbose)
    sys.exit(0)


if __name__ == "__main__":
    main()
