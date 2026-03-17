import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any
from ollama import Client


REQUIRED_FIELDS = {
    "description": str,
    "general_mood": str,
    "genre_tags": list,
    "lead_instrument": str,
    "accompaniment": str,
    "tempo_and_rhythm": str,
    "vocal_presence": str,
    "production_quality": str,
}


def slugify_model(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"


def read_dataset_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    maybe_json = text[start : end + 1]
    data = json.loads(maybe_json)
    if not isinstance(data, dict):
        raise ValueError("Model output JSON is not an object.")
    return data


def normalize_output(data: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, expected_type in REQUIRED_FIELDS.items():
        value = data.get(key)
        if expected_type is list:
            if isinstance(value, list):
                normalized[key] = [str(x) for x in value if str(x).strip() != ""]
            elif isinstance(value, str):
                normalized[key] = [value] if value.strip() else []
            else:
                normalized[key] = []
        else:
            normalized[key] = str(value).strip() if value is not None else ""
    return normalized


def ollama_chat_completion(client: Client, model: str, system_prompt: str, caption: str) -> str:
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Input caption: "{caption}"'},
        ],
        options={"temperature": 0.1},
    )
    return response["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["llama3.1:8b"],)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    csv_path = Path("data/dataset.csv")
    output_root = Path("data")
    prompt_path = Path("prompt.txt")
    ollama_url = "http://localhost:11434"
    client = Client(host=ollama_url)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    rows = read_dataset_rows(csv_path)
    total = len(rows)
    had_failures = False

    for model in args.models:
        model_slug = slugify_model(model)
        json_dir = output_root / f"json_{model_slug}"
        json_dir.mkdir(parents=True, exist_ok=True)

        ok = 0
        skipped = 0
        failed = 0

        print(f"\nModel: {model}")
        for idx, row in enumerate(rows, start=1):
            filename = str(row.get("filename", "")).strip()
            caption = str(row.get("caption", "")).strip()
            json_path = json_dir / f"{Path(filename).stem}.json"

            if args.skip_existing and json_path.exists():
                skipped += 1
                print(f"[{idx}/{total}] SKIP exists: {json_path.name}")
                continue

            try:
                content = ollama_chat_completion(
                    client=client,
                    model=model,
                    system_prompt=system_prompt,
                    caption=caption,
                )
                parsed = parse_json_object(content)
                normalized = normalize_output(parsed)
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(normalized, f, ensure_ascii=False, indent=2)
                ok += 1
                print(f"[{idx}/{total}] OK   {json_path.name}")
            except Exception as exc:
                failed += 1
                print(f"[{idx}/{total}] FAIL {type(exc).__name__}: {exc}", file=sys.stderr)

        print(f"JSON dir: {json_dir}")
        print(f"Generated: {ok}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")
    print("\nDone.")


if __name__ == "__main__":
    main()
