"""Create tabular proxy features from a text CSV.

This starter routine avoids transformer models. It extracts document-level text
features and optional TF-IDF/SVD components so you can test whether a tabular
representation is a useful baseline for your portfolio text problem.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def basic_text_features(text: str, keywords: list[str]) -> dict[str, float | int]:
    words = WORD_RE.findall(text.lower())
    unique_words = set(words)
    char_count = len(text)
    word_count = len(words)
    sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))
    keyword_counts = {f"keyword_{keyword}_count": text.lower().count(keyword) for keyword in keywords}

    return {
        "char_count": char_count,
        "word_count": word_count,
        "unique_word_count": len(unique_words),
        "unique_word_fraction": len(unique_words) / word_count if word_count else 0.0,
        "avg_word_length": float(np.mean([len(word) for word in words])) if words else 0.0,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": word_count / sentence_count,
        "digit_count": sum(ch.isdigit() for ch in text),
        "uppercase_count": sum(ch.isupper() for ch in text),
        "punctuation_count": sum((not ch.isalnum()) and (not ch.isspace()) for ch in text),
        **keyword_counts,
    }


def add_tfidf_svd_features(texts: pd.Series, components: int, max_features: int) -> pd.DataFrame:
    if components <= 0 or len(texts) < 3:
        return pd.DataFrame(index=texts.index)

    vectorizer = TfidfVectorizer(max_features=max_features, min_df=1, ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(texts.fillna(""))
    usable_components = min(components, max(1, min(tfidf.shape) - 1))
    svd = TruncatedSVD(n_components=usable_components, random_state=6320)
    reduced = svd.fit_transform(tfidf)
    return pd.DataFrame(
        reduced,
        columns=[f"tfidf_svd_{idx + 1:02d}" for idx in range(usable_components)],
        index=texts.index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tabular proxy features from text.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--keywords", default="", help="Comma-separated keywords to count.")
    parser.add_argument("--tfidf-components", type=int, default=20)
    parser.add_argument("--max-tfidf-features", type=int, default=2000)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    if args.text_column not in data.columns:
        raise ValueError(f"Missing text column: {args.text_column}")

    keywords = [word.strip().lower() for word in args.keywords.split(",") if word.strip()]
    texts = data[args.text_column].fillna("").astype(str)
    basic = pd.DataFrame([basic_text_features(text, keywords) for text in texts], index=data.index)
    svd = add_tfidf_svd_features(texts, args.tfidf_components, args.max_tfidf_features)

    output_parts = []
    if args.id_column:
        output_parts.append(data[[args.id_column]])
    output_parts.extend([basic, svd])
    if args.label_column:
        output_parts.append(data[[args.label_column]])

    features = pd.concat(output_parts, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Wrote {len(features)} rows and {len(features.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
