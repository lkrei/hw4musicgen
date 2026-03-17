import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Optional

import openai


logger = logging.getLogger("enrich_metadata_llm")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


SYSTEM_PROMPT = (
    "You are a music metadata extractor. "
    "You receive a free-form English text caption that describes a short music clip. "
    "Your task is to extract structured information and return it as a JSON object with exactly these fields:\n\n"
    "- description: a clean, concise one-sentence description of the music\n"
    "- general_mood: the overall emotional atmosphere (e.g. 'calm and melancholic', 'energetic and uplifting')\n"
    "- genre_tags: a JSON array of 1-4 genre strings (e.g. [\"Lo-Fi\", \"Hip Hop\"])\n"
    "- lead_instrument: the primary melodic instrument\n"
    "- accompaniment: supporting instruments and sound elements\n"
    "- tempo_and_rhythm: tempo feel and rhythmic character\n"
    "- vocal_presence: whether vocals are present and their role, or 'None' if purely instrumental\n"
    "- production_quality: production style and audio quality (e.g. 'lo-fi', 'polished studio')\n\n"
    "Rules:\n"
    "- Output ONLY valid JSON, no markdown fences, no extra text.\n"
    "- Do not invent instruments or genres absent from the caption.\n"
    "- Keep all values concise (one short phrase or sentence).\n"
    "- genre_tags must be a JSON array of strings, even if there is only one genre.\n"
)


def build_user_message(caption: str) -> str:
    return f"Caption:\n{caption.strip()}"


def call_llm(caption: str, client: openai.OpenAI, model: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(caption)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = response.choices[0].message.content
    return json.loads(content)


def process_metadata_csv(
    csv_path: Path,
    model: str,
    limit: Optional[int],
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    client = openai.OpenAI()

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    target = len(rows) if limit is None else min(len(rows), limit)
    logger.info("rows: %d target: %d", len(rows), target)

    processed = 0
    skipped = 0

    for row in rows:
        if limit is not None and processed >= limit:
            break

        audio_path = Path(row["path"])
        if not audio_path.exists():
            continue

        json_path = audio_path.with_suffix(".json")
        if json_path.exists():
            processed += 1
            skipped += 1
            continue

        caption = row["caption"]
        try:
            enriched = call_llm(caption=caption, client=client, model=model)
        except Exception as exc:
            logger.warning("llm error: %s: %s", audio_path.name, exc)
            continue

        if isinstance(enriched.get("genre_tags"), str):
            enriched["genre_tags"] = [enriched["genre_tags"]]

        with json_path.open("w") as jf:
            json.dump(enriched, jf, ensure_ascii=False, indent=2)

        processed += 1
        if processed % 50 == 0:
            logger.info("progress: %d/%d skipped: %d", processed, target, skipped)

    logger.info("done: processed: %d skipped: %d", processed, skipped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("missing OPENAI_API_KEY")

    process_metadata_csv(
        csv_path=Path(args.csv_path),
        model=args.model,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
