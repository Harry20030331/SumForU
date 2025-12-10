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
    # manually added
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

# ---------- Rating / Suitability cleaning helpers ----------

RATING_PREFIX_RE = re.compile(r"^\s*rating\s+([0-9]+(?:\.[0-9]+)?)\.\s*", flags=re.I)
SUITABILITY_RE = re.compile(r"Suitability\s*:\s*[0-9]+\s*/\s*10", flags=re.I)

def strip_rating_and_suitability(text: str) -> str:
    """
    Remove leading 'rating X.Y.' prefix and any 'Suitability: N/10' snippets,
    then strip whitespace. Used for text metrics (ROUGE/BERTScore/coverage...).
    """
    text = RATING_PREFIX_RE.sub("", text)
    text = SUITABILITY_RE.sub("", text)
    return text.strip()

# ---------- Text extraction helpers ----------

def extract_summary(pred: str) -> str:
    """
    Extract the Summary part from a full model output.

    - Remove <|im_end|>.
    - Keep the text between 'Summary:' and 'Suitability:' if present.
    - Then strip any rating/Suitability boilerplate.
    """
    pred = pred.replace("<|im_end|>", "").strip()
    m = re.search(r"Summary:(.*?)(Suitability:|$)", pred, flags=re.S)
    if not m:
        raw = pred.strip()
    else:
        raw = m.group(1).strip()
    return strip_rating_and_suitability(raw)

def load_predictions(path: Path) -> List[str]:
    """
    Load model outputs JSON (list[str]) and extract summaries.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [extract_summary(x) for x in data]

def generate_gt_as_model_outputs(gt_data: list) -> List[str]:
    """
    Generate GT-as-model outputs from ground truth data.
    """
    outputs = []
    for item in gt_data:
        refs = item.get("reference_output") or []
        if not refs:
            outputs.append("")
            continue

        ref = refs[0].strip()

        # 1) Search for 'rating X.Y' in the entire text
        m = re.search(r"rating\s+([0-9]+(?:\.[0-9]+)?)", ref, flags=re.I)
        if m:
            rating_str = m.group(1)
            try:
                rating = float(rating_str)
            except ValueError:
                rating = 5.0
        else:
            rating = 5.0

        # 2) Clean the text by removing rating and Suitability parts
        ref_clean = strip_rating_and_suitability(ref)

        # 3) Map 0-5 rating to 0-10 suitability (according to current evaluation logic: *2)
        suitability_10 = max(0, min(int(round(rating * 2)), 10))

        # 4) Build the output format
        out = f"Summary: {ref_clean}\n\nSuitability: {suitability_10}/10"

        outputs.append(out)

    return outputs
def load_references_and_ref_ratings(gt_data: list) -> Tuple[List[str], List[float]]:
    """
    From ground truth JSON (list[dict]), for each sample:

    - Take reference_output[0] as reference text.
    - For text metrics, clean rating/Suitability boilerplate.
    - For suitability metrics, extract 'rating X.Y' at the beginning of
      reference_output[0] as the GT rating (persona's true score).

    Returns:
      refs_clean: list[str]  -- for ROUGE/BERTScore etc.
      ref_ratings: list[float] -- for suitability vs rating metrics (NaN if missing).
    """
    refs_clean: List[str] = []
    ref_ratings: List[float] = []

    for item in gt_data:
        ref_list = item.get("reference_output") or []
        if not ref_list:
            refs_clean.append("")
            ref_ratings.append(float("nan"))
            continue

        raw_ref = ref_list[0].strip()

        m = RATING_PREFIX_RE.match(raw_ref)
        if m:
            try:
                rating_val = float(m.group(1))
            except ValueError:
                rating_val = float("nan")
        else:
            rating_val = float("nan")
        ref_ratings.append(rating_val)

        refs_clean.append(strip_rating_and_suitability(raw_ref))

    return refs_clean, ref_ratings

# ---------- persona / reviews text loader (for BERTScore) ----------

def load_persona_texts(gt_data: list) -> List[str]:
    """
    Load raw persona texts per sample for semantic similarity metrics.
    """
    personas: List[str] = []
    for item in gt_data:
        personas.append(item.get("persona", "").strip())
    return personas

def load_reviews_texts(gt_data: list) -> List[str]:
    """
    Load raw reviews texts per sample for semantic similarity metrics.
    """
    reviews: List[str] = []
    for item in gt_data:
        reviews.append(item.get("reviews", "").strip())
    return reviews

# ---------- ROUGE / BLEU / BERTScore ----------

def compute_rouge(preds: List[str], refs: List[str]) -> dict:
    rouge = evaluate.load("rouge")
    n = min(len(preds), len(refs))
    preds = preds[:n]
    refs = refs[:n]
    return rouge.compute(predictions=preds, references=refs, use_stemmer=True)

def compute_bleu(preds: List[str], refs: List[str]) -> float:
    bleu = evaluate.load("bleu")
    n = min(len(preds), len(refs))
    preds = preds[:n]
    refs = refs[:n]
    references_wrapped = [[r] for r in refs]
    result = bleu.compute(predictions=preds, references=references_wrapped)
    return float(result["bleu"])

def compute_bertscore(bertscore, preds: List[str], refs: List[str]) -> Tuple[float, float, float]:
    n = min(len(preds), len(refs))
    preds = preds[:n]
    refs = refs[:n]

    previous_level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        result = bertscore.compute(
            predictions=preds,
            references=refs,
            lang="en",
            model_type="distilbert-base-uncased",  # Use faster model
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
    if not preds:
        return 0.0
    total = len(preds)
    unique = len(set(s.strip() for s in preds))
    return unique / total

def compute_entropy(preds: List[str]) -> float:
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

def load_review_tokens(gt_data: list) -> List[set]:
    review_vocab_per_sample: List[set] = []
    for item in gt_data:
        reviews_text = item.get("reviews", "")
        tokens = set(simple_tokenize(reviews_text))
        review_vocab_per_sample.append(tokens)
    return review_vocab_per_sample

def load_persona_tokens(gt_data: list) -> List[set]:
    persona_vocab_per_sample: List[set] = []
    for item in gt_data:
        persona_text = item.get("persona", "")
        tokens = set(simple_tokenize(persona_text))
        persona_vocab_per_sample.append(tokens)
    return persona_vocab_per_sample

def compute_coverage(preds: List[str], vocab_per_sample: List[set]) -> float:
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

def align_and_filter(pred_scores: List[float], gt_scores: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(pred_scores), len(gt_scores))
    pred_arr = np.array(pred_scores[:n], dtype=float)
    gt_arr = np.array(gt_scores[:n], dtype=float)
    mask = ~np.isnan(pred_arr) & ~np.isnan(gt_arr)
    return pred_arr[mask], gt_arr[mask]

def compute_score_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
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
        description="Evaluate text quality/diversity + semantic similarity + coverage + suitability vs reference rating for multiple models on generalization tasks."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing generalization results, e.g., results/generalization/",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        required=True,
        help="Ground truth directory containing preprocessed generalization data, e.g., dataset/data/generalization/preprocessed/",
    )
    args = parser.parse_args()

    # Get categories from gt-dir
    gt_files = list(args.gt_dir.glob("preprocessed_*.json"))
    categories = [f.stem.replace("preprocessed_", "") for f in gt_files]

    # Preload BERTScore
    previous_level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    bertscore = evaluate.load("bertscore")
    hf_logging.set_verbosity(previous_level)

    for category in categories:
        print(f"\nProcessing category: {category}")
        gt_path = args.gt_dir / f"preprocessed_{category}.json"
        with gt_path.open("r", encoding="utf-8") as f:
            gt_data = json.load(f)

        refs, gt_ref_ratings = load_references_and_ref_ratings(gt_data)
        review_vocab_per_sample = load_review_tokens(gt_data)
        persona_vocab_per_sample = load_persona_tokens(gt_data)
        persona_texts = load_persona_texts(gt_data)
        reviews_texts = load_reviews_texts(gt_data)

        # Generate GT-as-model outputs
        gt_preds = generate_gt_as_model_outputs(gt_data)

        # Models: baseline, pe, sft, rl
        methods = {
            "gt": gt_preds,
        }
        category_dir = args.input_dir / category
        model_files = {
            "baseline": category_dir / "baseline.json",
            "pe": category_dir / "pe.json",
            "sft": category_dir / "sft.json",
            "rl": category_dir / "rl.json",
        }
        for name, path in model_files.items():
            if path.exists():
                methods[name] = path

        bert_f1_scores: dict[str, float] = {}
        text_quality_lines = []

        for name, data in methods.items():
            if isinstance(data, list):
                preds = [extract_summary(p) for p in data]
            else:
                preds = load_predictions(data)

            rouge_res = compute_rouge(preds, refs)
            bleu4 = compute_bleu(preds, refs)
            d2 = compute_distinct(preds, n=2)
            d3 = compute_distinct(preds, n=3)
            usr = compute_usr(preds)
            entr = compute_entropy(preds)
            rev_cov = compute_coverage(preds, review_vocab_per_sample)
            pers_cov = compute_coverage(preds, persona_vocab_per_sample)

            # summary vs reference_output
            print(f"Computing BERTScore for {name} vs reference...")
            try:
                _, ref_r, ref_f1 = compute_bertscore(bertscore, preds, refs)
            except Exception as e:
                print(f"Error in BERTScore for {name} vs reference: {e}")
                ref_r, ref_f1 = float("nan"), float("nan")
            bert_f1_scores[name] = ref_f1

            # summary vs reviews: RevBS-P
            print(f"Computing BERTScore for {name} vs reviews...")
            try:
                rev_p, _, _ = compute_bertscore(bertscore, preds, reviews_texts)
            except Exception as e:
                print(f"Error in BERTScore for {name} vs reviews: {e}")
                rev_p = float("nan")

            # summary vs persona: PersBS-R
            print(f"Computing BERTScore for {name} vs persona...")
            try:
                _, pers_r, _ = compute_bertscore(bertscore, preds, persona_texts)
            except Exception as e:
                print(f"Error in BERTScore for {name} vs persona: {e}")
                pers_r = float("nan")

            line = "{:<10} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:9.4f} {:9.4f} {:9.4f}".format(
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
                rev_cov,
                pers_cov,
                ref_r,
                rev_p,
                pers_r,
            )
            text_quality_lines.append(line)

        print("=== Text quality, diversity, semantic similarity, and coverage ===")
        print(
            "{:<10} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9} {:>9} {:>9}".format(
                "method",
                "r1", "r2", "rL", "rLsum",
                "bleu4",
                "D-2", "D-3", "USR", "ENTR",
                "RevCov", "PersCov",
                "RefBS-R", "RevBS-P", "PersBS-R",
            )
        )

        for line in text_quality_lines:
            print(line)

        print("\n=== Suitability vs reference rating (0-5 scale) ===")
        print(
            "{:<10} {:>7} {:>7} {:>9} {:>9} {:>10} {:>12} {:>10} {:>14}".format(
                "method", "MAE", "MSE", "Pearson", "Spearman",
                "ExactAcc", "Within1Acc", "MacroF1", "BalancedAcc"
            )
        )

        suitability_lines = []

        for name, data in methods.items():
            if isinstance(data, list):
                pred_scores_raw = [extract_suitability(pred) for pred in data]
                pred_scores_raw = [raw / 2.0 if raw is not None else float("nan") for raw in pred_scores_raw]
            else:
                pred_scores_raw = load_suitability_scores(data)
            pred_arr, gt_arr = align_and_filter(pred_scores_raw, gt_ref_ratings)
            score_metrics = compute_score_metrics(pred_arr, gt_arr)
            cls_metrics = compute_classification_metrics(pred_arr, gt_arr)

            line = "{:<10} {:7.4f} {:7.4f} {:9.4f} {:9.4f} {:10.4f} {:12.4f} {:10.4f} {:14.4f}".format(
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
            suitability_lines.append(line)

        for line in suitability_lines:
            print(line)

        # Save results to JSON
        results = {
            "category": category,
            "text_quality": {},
            "suitability": {},
        }
        for i, name in enumerate(methods.keys()):
            results["text_quality"][name] = {
                "rouge1": float(text_quality_lines[i].split()[1]),
                "rouge2": float(text_quality_lines[i].split()[2]),
                "rougeL": float(text_quality_lines[i].split()[3]),
                "rougeLsum": float(text_quality_lines[i].split()[4]),
                "bleu4": float(text_quality_lines[i].split()[5]),
                "D-2": float(text_quality_lines[i].split()[6]),
                "D-3": float(text_quality_lines[i].split()[7]),
                "USR": float(text_quality_lines[i].split()[8]),
                "ENTR": float(text_quality_lines[i].split()[9]),
                "RevCov": float(text_quality_lines[i].split()[10]),
                "PersCov": float(text_quality_lines[i].split()[11]),
                "RefBS-R": float(text_quality_lines[i].split()[12]),
                "RevBS-P": float(text_quality_lines[i].split()[13]),
                "PersBS-R": float(text_quality_lines[i].split()[14]),
            }
            results["suitability"][name] = {
                "MAE": float(suitability_lines[i].split()[1]),
                "MSE": float(suitability_lines[i].split()[2]),
                "Pearson": float(suitability_lines[i].split()[3]),
                "Spearman": float(suitability_lines[i].split()[4]),
                "ExactAcc": float(suitability_lines[i].split()[5]),
                "Within1Acc": float(suitability_lines[i].split()[6]),
                "MacroF1": float(suitability_lines[i].split()[7]),
                "BalancedAcc": float(suitability_lines[i].split()[8]),
            }

        output_path = category_dir / "rule_metrics.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
