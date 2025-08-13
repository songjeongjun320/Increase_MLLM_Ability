#!/usr/bin/env python3
"""
Generate ToW-augmented KLUE dataset in batches
================================================

This script reads EVALUATION/klue_all.json, splits it into N batches, and for each
entry generates an English ToW token inserted into the Korean sentence, producing
records aligned with the schema demonstrated in `klue_tow_manual_samples.json`.

Output:
- EVALUATION/klue_tow_all.part_XX.json (20 parts by default)
- EVALUATION/klue_tow_all.json (merged array of all parts)

Notes:
- ToW content is generated in English using heuristics from the existing
  KoreanLinguisticAnalyzer and templates.
- If no linguistic "challenge" is detected, a safe fallback ToW token is created
  and inserted before the final word of the sentence.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Logging setup
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("klue_tow_batch")

# Reuse analyzer and classifier from existing code
try:
    # These exist in the repo and provide robust challenge detection and templated reasoning
    from generate_complete_tow_dataset import KoreanLinguisticAnalyzer, classify_word_category
except Exception as e:  # Fallback if import path issues occur
    logger.error("Failed to import KoreanLinguisticAnalyzer from generate_complete_tow_dataset.py: %s", e)
    raise


@dataclass
class TOWEntry:
    doc_id: str
    original_text: str
    augmented_text: str
    tow_tokens: List[str]
    tow_count: int
    predicted_word: str
    difficulty_markers: List[str]
    word_category: str
    prediction_challenge: str
    source: str
    story_id: str
    sentence_id: int


def _find_last_word_with_position(text: str) -> Optional[Dict[str, int]]:
    """Find the last plausible Korean word (or token) and return its start index and value."""
    # Match a sequence of Korean letters/numbers/some punctuation; grab the last match
    # This is intentionally generous; the analyzer is preferred when available
    pattern = re.compile(r"([\w가-힣A-Za-z%~:]+)(?=[^\w가-힣A-Za-z%~:]*$)")
    m = pattern.search(text)
    if not m:
        return None
    return {"word": m.group(1), "pos": m.start(1)}


def _insert_tow_at(text: str, pos: int, token: str) -> str:
    return text[:pos] + token + text[pos:]


def process_chunk(
    analyzer: KoreanLinguisticAnalyzer,
    items: List[Dict],
    story_id: str = "klue_all",
) -> List[TOWEntry]:
    results: List[TOWEntry] = []
    for item in items:
        sid = int(item.get("id", 0))
        sentence = str(item.get("sentence", "")).strip()
        if not sentence:
            # Skip empty entries, but keep structure
            results.append(
                TOWEntry(
                    doc_id=f"klue_{sid}_auto_1",
                    original_text="",
                    augmented_text="",
                    tow_tokens=[],
                    tow_count=0,
                    predicted_word="",
                    difficulty_markers=[],
                    word_category="trivial",
                    prediction_challenge="none",
                    source="klue",
                    story_id=story_id,
                    sentence_id=sid,
                )
            )
            continue

        # Analyze challenges using existing rule-based analyzer
        challenges = analyzer.analyze_text_challenges(sentence)

        if challenges:
            # Choose the earliest high-priority challenge to insert before its word
            selected = analyzer.select_optimal_points(challenges, max_points=1)[0]
            word = selected["word"]
            position = selected["position"]

            # Generate English ToW reasoning (already formatted with <ToW>..</ToW>)
            tow_token = analyzer.generate_tow_reasoning(
                selected,
                context_before=sentence[:position],
                context_after=sentence[position + len(word) :],
            )

            augmented = _insert_tow_at(sentence, position, tow_token)
            entry = TOWEntry(
                doc_id=f"klue_{sid}_auto_1",
                original_text=sentence,
                augmented_text=augmented,
                tow_tokens=[tow_token],
                tow_count=1,
                predicted_word=word,
                difficulty_markers=[selected["type"]],
                word_category=classify_word_category(selected),
                prediction_challenge=selected["subtype"],
                source="klue",
                story_id=story_id,
                sentence_id=sid,
            )
            results.append(entry)
        else:
            # Fallback: insert a generic ToW token before the last word
            fallback = _find_last_word_with_position(sentence)
            if fallback:
                word = fallback["word"]
                position = fallback["pos"]
                tow_token = (
                    f"<ToW>The context suggests that the following word '{word}' logically completes the sentence in Korean. This is a simple continuity prediction.</ToW>"
                )
                augmented = _insert_tow_at(sentence, position, tow_token)
                entry = TOWEntry(
                    doc_id=f"klue_{sid}_auto_1",
                    original_text=sentence,
                    augmented_text=augmented,
                    tow_tokens=[tow_token],
                    tow_count=1,
                    predicted_word=word,
                    difficulty_markers=["contextual_continuation"],
                    word_category="soft",
                    prediction_challenge="simple_continuation",
                    source="klue",
                    story_id=story_id,
                    sentence_id=sid,
                )
                results.append(entry)
            else:
                # Could not even find a last word; keep empty ToW info
                entry = TOWEntry(
                    doc_id=f"klue_{sid}_auto_1",
                    original_text=sentence,
                    augmented_text=sentence,
                    tow_tokens=[],
                    tow_count=0,
                    predicted_word="",
                    difficulty_markers=[],
                    word_category="trivial",
                    prediction_challenge="none",
                    source="klue",
                    story_id=story_id,
                    sentence_id=sid,
                )
                results.append(entry)

    return results


def save_json(path: Path, records: List[TOWEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Batch-generate ToW tokens for KLUE dataset")
    parser.add_argument("--input", default="EVALUATION/klue_all.json", help="Path to klue_all.json")
    parser.add_argument("--output", default="EVALUATION/klue_tow_all.json", help="Final merged output path")
    parser.add_argument("--parts", type=int, default=20, help="Number of parts to split into")
    parser.add_argument("--part-dir", default="EVALUATION", help="Directory to write part files into")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", args.input)
        raise SystemExit(1)

    logger.info("Loading KLUE dataset: %s", input_path)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error("Input JSON must be a list of objects with 'id' and 'sentence'")
        raise SystemExit(1)

    total = len(data)
    parts = max(1, int(args.parts))
    chunk_size = math.ceil(total / parts)
    logger.info("Total entries: %d | parts: %d | chunk_size: %d", total, parts, chunk_size)

    analyzer = KoreanLinguisticAnalyzer()

    part_paths: List[Path] = []
    all_results: List[TOWEntry] = []

    for i in range(parts):
        start = i * chunk_size
        end = min(total, (i + 1) * chunk_size)
        if start >= end:
            break
        chunk = data[start:end]
        logger.info("Processing part %d (%d..%d)", i + 1, start, end - 1)

        results = process_chunk(analyzer, chunk, story_id="klue_all")

        part_path = Path(args.part_dir) / f"klue_tow_all.part_{i+1:02d}.json"
        save_json(part_path, results)
        part_paths.append(part_path)
        all_results.extend(results)

    # Merge into final file
    output_path = Path(args.output)
    logger.info("Saving merged output: %s (records=%d)", output_path, len(all_results))
    save_json(output_path, all_results)
    logger.info("Done. Part files: %s", ", ".join(str(p) for p in part_paths))


if __name__ == "__main__":
    main()


