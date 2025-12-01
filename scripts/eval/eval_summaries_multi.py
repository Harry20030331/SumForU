import argparse
import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import evaluate
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from transformers.utils import logging as hf_logging


# ---------- Simple tokenizer / stopwords ----------

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "when", "while",
    "of", "for", "in", "on", "at", "to", "from", "by", "with", "about", "as",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "we", "our", "you", "your", "i", "me",
    #manully added
    "really", "very", "quite", "just", "still", "even", "also",
}

def simple_tokenize(text: str) -> List[str]:
    """
    Simple whitespace + punctuation tokenization to get lowercase content words.
    """
    text = text.lower()
    trans = str.maketrans({ch: " " for ch in string.punctuation})
    text = text.translate(trans)
    tokens = [t for t in text.split() if t and t not in STOPWORDS]
    return tokens


# ---------- Text extraction helpers ----------

def extract_summary(pred: str) -> str:
    """
    Extract the Summary part from a full model output.
    - Remove <|im_end|>.
    - Keep the text between 'Summary:' and 'Suitability:' if present.
    """
    pred = pred.replace("<|im_end|>", "").strip()
    m = re.search(r"Summary:(.*?)(Suitability:|$)", pred, flags=re.S)
    if not m:
        return pred.strip()
    return m.group(1).strip()


def load_predictions(path: Path) -> List[str]:
    """
    Load model outputs JSON (list[str]) and extract summaries.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [extract_summary(x) for x in data]


def load_references(gt_path: Path) -> List[str]:
    """
    Load ground truth JSON (list[dict]) and take reference_output[0] as reference text.
    """
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    refs = []
    for item in data:
        ref_list = item.get("reference_output") or []
        if not ref_list:
            refs.append("")
        else:
            refs.append(ref_list[0].strip())
    return refs


# ---------- ROUGE / BLEU / BERTScore ----------

def compute_rouge(preds: List[str], refs: List[str]) -> dict:
    """
    Compute ROUGE-1/2/L/Lsum using HuggingFace evaluate.
    """
    rouge = evaluate.load("rouge")
    n = min(len(preds), len(refs))
    preds = preds[:n]
    refs = refs[:n]
    return rouge.compute(predictions=preds, references=refs, use_stemmer=True)


def compute_bleu(preds: List[str], refs: List[str]) -> float:
    """
    Compute BLEU (default BLEU-4) using HuggingFace evaluate.
    Mostly a supplementary signal for this task.
    """
    bleu = evaluate.load("bleu")
    n = min(len(preds), len(refs))
    preds = preds[:n]
    refs = refs[:n]
    references_wrapped = [[r] for r in refs]
    result = bleu.compute(predictions=preds, references=references_wrapped)
    return float(result["bleu"])


def compute_bertscore(preds: List[str], refs: List[str]) -> Tuple[float, float, float]:
    """
    Compute BERTScore (precision, recall, F1) using HuggingFace evaluate.
    We average over all examples.
    """
    bertscore = evaluate.load("bertscore")
    n = min(len(preds), len(refs))
    preds = preds[:n]
    refs = refs[:n]

    # Temporarily silence transformers info logs (e.g., pooler init).
    previous_level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        result = bertscore.compute(
            predictions=preds,
            references=refs,
            lang="en",
            rescale_with_baseline=False,
        )
    finally:
        hf_logging.set_verbosity(previous_level)

    p = float(np.mean(result["precision"]))
    r = float(np.mean(result["recall"]))
    f1 = float(np.mean(result["f1"]))
    return p, r, f1


# ---------- Diversity metrics: Distinct / USR / ENTR ----------

def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_distinct(preds: List[str], n: int) -> float:
    """
    Compute token-level Distinct-n:
    Distinct-n = (# unique n-grams) / (# total n-grams).
    """
    all_ngrams: List[Tuple[str, ...]] = []
    for sent in preds:
        tokens = sent.strip().split()
        if len(tokens) < n:
            continue
        all_ngrams.extend(_get_ngrams(tokens, n))

    if not all_ngrams:
        return 0.0

    total = len(all_ngrams)
    unique = len(set(all_ngrams))
    return unique / total


def compute_usr(preds: List[str]) -> float:
    """
    Compute Unique Sentence Ratio (USR):
    USR = (# unique sentences) / (# total sentences).
    """
    if not preds:
        return 0.0
    total = len(preds)
    unique = len(set(s.strip() for s in preds))
    return unique / total


def compute_entropy(preds: List[str]) -> float:
    """
    Compute unigram entropy ENTR over all generated summaries:
    ENTR = -sum_w p(w) * log(p(w)).
    """
    tokens: List[str] = []
    for sent in preds:
        tokens.extend(sent.strip().split())

    if not tokens:
        return 0.0

    counts = Counter(tokens)
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p + 1e-12)
    return entropy


# ---------- Coverage metrics (review / persona) ----------

def load_review_tokens(gt_path: Path) -> List[set]:
    """
    For each sample, extract a set of content-word tokens from its reviews field.
    """
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    review_vocab_per_sample: List[set] = []
    for item in data:
        reviews_text = item.get("reviews", "")
        tokens = set(simple_tokenize(reviews_text))
        review_vocab_per_sample.append(tokens)
    return review_vocab_per_sample


def load_persona_tokens(gt_path: Path) -> List[set]:
    """
    For each sample, extract a set of content-word tokens from its persona field.
    """
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    persona_vocab_per_sample: List[set] = []
    for item in data:
        persona_text = item.get("persona", "")
        tokens = set(simple_tokenize(persona_text))
        persona_vocab_per_sample.append(tokens)
    return persona_vocab_per_sample


def compute_coverage(preds: List[str], vocab_per_sample: List[set]) -> float:
    """
    Compute average token-level coverage:
    For each sample:
      cov_i = (# summary tokens that appear in vocab_per_sample[i]) / (# summary tokens)
    Return the mean cov_i over all valid samples.
    """
    n = min(len(preds), len(vocab_per_sample))
    coverages: List[float] = []
    for i in range(n):
        summary_tokens = simple_tokenize(preds[i])
        if not summary_tokens:
            continue
        vocab = vocab_per_sample[i]
        hit = sum(1 for w in summary_tokens if w in vocab)
        cov = hit / (len(summary_tokens) + 1e-12)
        coverages.append(cov)
    if not coverages:
        return 0.0
    return float(np.mean(coverages))


# ---------- Suitability vs rating helpers ----------

def extract_suitability(pred: str) -> float | None:
    """
    Extract suitability score from a model output string.
    Expect patterns like: 'Suitability: 8/10'.
    Return the raw 0-10 integer as float, or None if not found.
    """
    pred = pred.replace("<|im_end|>", "")
    m = re.search(r"Suitability\s*:\s*([0-9]+)\s*/\s*10", pred)
    if not m:
        return None
    try:
        val = float(m.group(1))
        return val
    except ValueError:
        return None


def load_suitability_scores(path: Path) -> List[float]:
    """
    Load model outputs JSON (list[str]) and extract suitability scores.
    Convert each to 0-5 scale by dividing by 2.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    scores: List[float] = []
    for pred in data:
        raw = extract_suitability(pred)
        if raw is None:
            scores.append(float("nan"))
        else:
            scores.append(raw / 2.0)
    return scores


def extract_ratings_from_reviews(reviews_text: str) -> List[float]:
    """
    Extract all 'rating x.y' patterns from the concatenated reviews text.
    Return list of floats (e.g., [5.0, 3.0, 4.0]).
    """
    matches = re.findall(r"rating\s+([0-9]+(?:\.[0-9]+)?)", reviews_text)
    ratings: List[float] = []
    for m in matches:
        try:
            ratings.append(float(m))
        except ValueError:
            continue
    return ratings


def load_avg_ratings(gt_path: Path) -> List[float]:
    """
    For each sample in ground truth JSON, compute the average rating
    over all 'rating x.y' occurrences in its 'reviews' field.
    If no rating is found, use NaN.
    """
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    avg_ratings: List[float] = []
    for item in data:
        reviews_text = item.get("reviews", "")
        ratings = extract_ratings_from_reviews(reviews_text)
        if not ratings:
            avg_ratings.append(float("nan"))
        else:
            avg_ratings.append(float(sum(ratings) / len(ratings)))
    return avg_ratings


def align_and_filter(pred_scores: List[float], gt_scores: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align prediction and ground truth scores by index, and drop any pair
    where either side is NaN.
    """
    n = min(len(pred_scores), len(gt_scores))
    pred_arr = np.array(pred_scores[:n], dtype=float)
    gt_arr = np.array(gt_scores[:n], dtype=float)

    mask = ~np.isnan(pred_arr) & ~np.isnan(gt_arr)
    return pred_arr[mask], gt_arr[mask]


def compute_score_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute scalar metrics between suitability (0-5) and avg rating (0-5):
    - MAE
    - MSE
    - Pearson correlation
    - Spearman correlation
    - exact_match_acc: round(pred) == round(gt)
    - within_1_acc: |round(pred) - round(gt)| <= 1
    """
    if pred.size == 0:
        return {
            "mae": math.nan,
            "mse": math.nan,
            "pearson": math.nan,
            "spearman": math.nan,
            "exact_acc": math.nan,
            "within1_acc": math.nan,
        }

    diff = pred - gt
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))

    try:
        pearson, _ = pearsonr(pred, gt)
    except Exception:
        pearson = math.nan

    try:
        spearman, _ = spearmanr(pred, gt)
    except Exception:
        spearman = math.nan

    pred_round = np.rint(pred)
    gt_round = np.rint(gt)

    exact_acc = float(np.mean(pred_round == gt_round))
    within1_acc = float(np.mean(np.abs(pred_round - gt_round) <= 1.0))

    return {
        "mae": mae,
        "mse": mse,
        "pearson": float(pearson),
        "spearman": float(spearman),
        "exact_acc": exact_acc,
        "within1_acc": within1_acc,
    }


def compute_classification_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Treat ratings as 5-class labels and compute:
    - macro_f1
    - balanced_accuracy

    Labels:
      gt_label   = round(gt) clipped to [1,5]
      pred_label = round(pred) clipped to [1,5] (0 is mapped up to 1)

    This version keeps the original metric semantics and only suppresses
    sklearn's warnings about unseen classes.
    """
    if pred.size == 0:
        return {"macro_f1": math.nan, "balanced_acc": math.nan}

    gt_label = np.rint(gt)
    pred_label = np.rint(pred)

    gt_label = np.clip(gt_label, 1, 5).astype(int)
    pred_label = np.clip(pred_label, 1, 5).astype(int)

    all_classes = np.array([1, 2, 3, 4, 5], dtype=int)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)

            macro_f1 = f1_score(
                gt_label,
                pred_label,
                labels=all_classes,
                average="macro",
                zero_division=0,
            )
            bal_acc = balanced_accuracy_score(
                gt_label,
                pred_label,
                adjusted=False,
            )
    except Exception:
        macro_f1 = math.nan
        bal_acc = math.nan

    return {"macro_f1": float(macro_f1), "balanced_acc": float(bal_acc)}


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate text quality/diversity + semantic similarity + coverage + suitability vs rating for multiple models."
    )
    parser.add_argument(
        "--gt-path",
        type=Path,
        required=True,
        help="Ground truth JSON path, e.g. dataset/data/raw/v1_test_preprocessed.json",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        required=True,
        help="Baseline outputs JSON",
    )
    parser.add_argument(
        "--pe-path",
        type=Path,
        required=True,
        help="PE model outputs JSON",
    )
    parser.add_argument(
        "--sft-path",
        type=Path,
        required=True,
        help="SFT model outputs JSON",
    )
    parser.add_argument(
        "--rl-path",
        type=Path,
        required=True,
        help="RL model outputs JSON",
    )
    args = parser.parse_args()

    refs = load_references(args.gt_path)
    gt_avg_ratings = load_avg_ratings(args.gt_path)
    review_vocab_per_sample = load_review_tokens(args.gt_path)
    persona_vocab_per_sample = load_persona_tokens(args.gt_path)

    methods = {
        "baseline": args.baseline_path,
        "pe": args.pe_path,
        "sft": args.sft_path,
        "rl": args.rl_path,
    }

    print("=== Text quality, diversity, semantic similarity, and coverage ===\n")
    print(
        "{:<10} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}".format(
            "method",
            "r1", "r2", "rL", "rLsum",
            "bleu4",
            "D-2", "D-3", "USR", "ENTR",
            "BS-P", "BS-R",
            "RevCov", "PersCov",
        )
    )

    bert_scores: dict[str, Tuple[float, float, float]] = {}

    for name, path in methods.items():
        preds = load_predictions(path)
        rouge_res = compute_rouge(preds, refs)
        bleu4 = compute_bleu(preds, refs)
        d2 = compute_distinct(preds, n=2)
        d3 = compute_distinct(preds, n=3)
        usr = compute_usr(preds)
        entr = compute_entropy(preds)

        try:
            bert_p, bert_r, bert_f = compute_bertscore(preds, refs)
        except Exception:
            bert_p, bert_r, bert_f = float("nan"), float("nan"), float("nan")
        bert_scores[name] = (bert_p, bert_r, bert_f)

        rev_cov = compute_coverage(preds, review_vocab_per_sample)
        pers_cov = compute_coverage(preds, persona_vocab_per_sample)

        print(
            "{:<10} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f}".format(
                name,
                rouge_res.get("rouge1", 0.0),
                rouge_res.get("rouge2", 0.0),
                rouge_res.get("rougeL", 0.0),
                rouge_res.get("rougeLsum", 0.0),
                bleu4,
                d2,
                d3,
                usr,
                entr,
                bert_p,
                bert_r,
                rev_cov,
                pers_cov,
            )
        )

    print("\nBERTScore F1 per method:")
    for name in methods:
        _, _, bert_f = bert_scores[name]
        print(f"  {name:<10} BERTScore-F1 = {bert_f:.4f}")

    print("\n=== Suitability vs average rating (0-5 scale) ===\n")
    print(
        "{:<10} {:>7} {:>7} {:>9} {:>9} {:>10} {:>12} {:>10} {:>14}".format(
            "method", "MAE", "MSE", "Pearson", "Spearman",
            "ExactAcc", "Within1Acc", "MacroF1", "BalancedAcc"
        )
    )

    for name, path in methods.items():
        pred_scores_raw = load_suitability_scores(path)
        pred_arr, gt_arr = align_and_filter(pred_scores_raw, gt_avg_ratings)

        score_metrics = compute_score_metrics(pred_arr, gt_arr)
        cls_metrics = compute_classification_metrics(pred_arr, gt_arr)

        print(
            "{:<10} {:7.4f} {:7.4f} {:9.4f} {:9.4f} {:10.4f} {:12.4f} {:10.4f} {:14.4f}".format(
                name,
                score_metrics["mae"],
                score_metrics["mse"],
                score_metrics["pearson"],
                score_metrics["spearman"],
                score_metrics["exact_acc"],
                score_metrics["within1_acc"],
                cls_metrics["macro_f1"],
                cls_metrics["balanced_acc"],
            )
        )


if __name__ == "__main__":
    main()
