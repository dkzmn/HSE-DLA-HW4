import argparse
import json
from pathlib import Path
from typing import Any
import torch
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen
from audiocraft.utils import export


def prompt_to_text(prompt: dict[str, Any]) -> str:
    fields = [
        ("description", prompt["description"]),
        ("general_mood", prompt["general_mood"]),
        ("genre_tags", ", ".join(prompt["genre_tags"])),
        ("lead_instrument", prompt["lead_instrument"]),
        ("accompaniment", prompt["accompaniment"]),
        ("tempo_and_rhythm", prompt["tempo_and_rhythm"]),
        ("vocal_presence", prompt["vocal_presence"]),
        ("production_quality", prompt["production_quality"]),
    ]
    return ". ".join([f"{k}: {v}" for k, v in fields])


def load_prompts(prompts_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    prompt_files = sorted(prompts_dir.glob("*.json"))
    prompts = []
    for path in prompt_files:
        with path.open("r", encoding="utf-8") as f:
            prompt = json.load(f)
        prompts.append((path.stem, prompt))
    return prompts


def resolve_model_path(model_path: str, export_dir: Path) -> str:
    p = Path(model_path)
    if not p.exists():
        return model_path

    if p.is_dir():
        lm = p / "state_dict.bin"
        comp = p / "compression_state_dict.bin"
        if lm.exists() and comp.exists():
            return str(p)
        raise ValueError(
            f"Directory exists but missing exported files: {p}. "
            "Expected state_dict.bin and compression_state_dict.bin."
        )

    if p.suffix == ".th":
        export_dir.mkdir(parents=True, exist_ok=True)
        lm_path = export_dir / "state_dict.bin"
        comp_path = export_dir / "compression_state_dict.bin"
        if not lm_path.exists():
            export.export_lm(p, lm_path)
        if not comp_path.exists():
            export.export_pretrained_compression_model("facebook/encodec_32khz", comp_path)
        return str(export_dir)

    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("../generated_wav"))
    parser.add_argument("--duration", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--auto-export-dir", type=Path, default=Path("../auto_export_model"))
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    prompts = load_prompts(args.prompts_dir)

    resolved_model_path = resolve_model_path(args.model_path, args.auto_export_dir)
    model = MusicGen.get_pretrained(resolved_model_path, device=args.device)
    safe_stride = max(1.0, min(5.0, float(model.max_duration) - 0.5))
    safe_duration = min(float(args.duration), float(model.max_duration))
    model.set_generation_params(
        duration=safe_duration,
        top_k=args.top_k,
        top_p=0.0,
        temperature=args.temperature,
        extend_stride=safe_stride,
    )

    for stem_name, prompt in prompts:
        prompt_text = prompt_to_text(prompt)
        wav = model.generate([prompt_text], progress=True)[0].cpu()
        stem = args.output_dir / stem_name
        audio_write(stem, wav, model.sample_rate, strategy="loudness")
        print(f"[OK] {stem_name} generated")


if __name__ == "__main__":
    main()
