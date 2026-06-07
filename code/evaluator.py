"""
evaluator.py
============

Compute evaluation metrics by comparing agent predictions against
sample_support_tickets.csv ground truth.

METRICS:
    status_accuracy     — replied/escalated classification accuracy
    status_f1_replied   — F1 for the "replied" class
    status_f1_escalated — F1 for the "escalated" class
    request_type_accuracy — 4-class accuracy
    product_area_accuracy — top-1 product_area match (case-insensitive)
    false_reply_rate    — escalated ground-truth rows we replied to (dangerous)
    over_escalation_rate — replied ground-truth rows we escalated (wasteful)

COLUMN MAPPING:
    Ground truth (sample_support_tickets.csv):
        Status → "Replied" / "Escalated"  (title-case)
        Product Area, Request Type
    Predictions (output of agent.run_agent on sample set):
        status → "replied" / "escalated"  (lower-case)
        product_area, request_type

USAGE:
    Called by main.py eval command.
    evaluate(predictions_df, ground_truth_df) → list[(name, value, note)]
"""

from __future__ import annotations

import re
from typing import Union

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm_status(val: str) -> str:
    v = str(val).strip().lower()
    if v in {"replied", "reply", "resolved"}:
        return "replied"
    if v in {"escalated", "escalate"}:
        return "escalated"
    return v


def _norm_request_type(val: str) -> str:
    v = str(val).strip().lower().replace(" ", "_")
    mapping = {
        "product_issues": "product_issue",
        "productissue":   "product_issue",
        "feature_requests": "feature_request",
        "featurerequest":   "feature_request",
        "bugs":    "bug",
        "defect":  "bug",
        "invalid_request": "invalid",
        "out_of_scope":    "invalid",
    }
    return mapping.get(v, v)


def _norm_area(val: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", str(val).strip().lower())


# ─────────────────────────────────────────────────────────────────────────────
# BINARY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _accuracy(pred: list, true: list) -> float:
    if not true:
        return 0.0
    return sum(p == t for p, t in zip(pred, true)) / len(true)


def _precision_recall_f1(pred: list, true: list, pos_label: str) -> tuple[float, float, float]:
    tp = sum(p == pos_label and t == pos_label for p, t in zip(pred, true))
    fp = sum(p == pos_label and t != pos_label for p, t in zip(pred, true))
    fn = sum(p != pos_label and t == pos_label for p, t in zip(pred, true))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    predictions:  pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> list[tuple[str, float, str]]:
    """
    Compare agent predictions against labeled ground truth.

    Args:
        predictions:  DataFrame with columns: status, product_area, request_type
                      (output of agent.run_agent on sample set)
        ground_truth: DataFrame with columns: Status, Product Area, Request Type
                      (sample_support_tickets.csv)

    Returns:
        List of (metric_name, score_0_to_1, note_string) tuples.
        Consumed by main.py eval for display.
    """
    n = min(len(predictions), len(ground_truth))
    if n == 0:
        return [("ERROR", 0.0, "no rows to evaluate")]

    pred = predictions.iloc[:n].reset_index(drop=True)
    gt   = ground_truth.iloc[:n].reset_index(drop=True)

    # ── Normalise ─────────────────────────────────────────────────────────────
    pred_status  = [_norm_status(v)       for v in pred.get("status",       [""] * n)]
    gt_status    = [_norm_status(v)       for v in gt.get("Status",         [""] * n)]

    pred_rtype   = [_norm_request_type(v) for v in pred.get("request_type", [""] * n)]
    gt_rtype     = [_norm_request_type(v) for v in gt.get("Request Type",   [""] * n)]

    pred_area    = [_norm_area(v)         for v in pred.get("product_area",  [""] * n)]
    gt_area      = [_norm_area(v)         for v in gt.get("Product Area",    [""] * n)]

    # ── Status metrics ────────────────────────────────────────────────────────
    status_acc = _accuracy(pred_status, gt_status)

    _, _, f1_replied   = _precision_recall_f1(pred_status, gt_status, "replied")
    _, _, f1_escalated = _precision_recall_f1(pred_status, gt_status, "escalated")

    n_gt_esc  = sum(1 for t in gt_status if t == "escalated")
    n_gt_repl = sum(1 for t in gt_status if t == "replied")

    false_reply_rate = (
        sum(p == "replied" and t == "escalated" for p, t in zip(pred_status, gt_status))
        / n_gt_esc if n_gt_esc > 0 else 0.0
    )
    over_esc_rate = (
        sum(p == "escalated" and t == "replied" for p, t in zip(pred_status, gt_status))
        / n_gt_repl if n_gt_repl > 0 else 0.0
    )

    # ── Request type accuracy ─────────────────────────────────────────────────
    rtype_acc = _accuracy(pred_rtype, gt_rtype)

    # ── Product area accuracy — skip blank ground truth ──────────────────────
    area_pairs = [(p, t) for p, t in zip(pred_area, gt_area) if t]
    area_acc = (
        sum(p == t for p, t in area_pairs) / len(area_pairs)
        if area_pairs else 0.0
    )

    # ── Build return list ─────────────────────────────────────────────────────
    results: list[tuple[str, float, str]] = [
        (
            "Status Accuracy",
            status_acc,
            f"{int(status_acc*n)}/{n} correct",
        ),
        (
            "Status F1 — Replied",
            f1_replied,
            f"precision/recall/F1 for 'replied' class",
        ),
        (
            "Status F1 — Escalated",
            f1_escalated,
            f"precision/recall/F1 for 'escalated' class",
        ),
        (
            "Request Type Accuracy",
            rtype_acc,
            f"{int(rtype_acc*n)}/{n} correct",
        ),
        (
            "Product Area Accuracy",
            area_acc,
            f"{int(area_acc*len(area_pairs))}/{len(area_pairs)} with labelled area",
        ),
        (
            "False Reply Rate  ↓",
            1.0 - false_reply_rate,   # invert: higher=better for display
            f"{false_reply_rate:.1%} of escalation-worthy tickets wrongly replied to",
        ),
        (
            "Over-Escalation Rate ↓",
            1.0 - over_esc_rate,      # invert: higher=better for display
            f"{over_esc_rate:.1%} of simple tickets unnecessarily escalated",
        ),
    ]

    return results
