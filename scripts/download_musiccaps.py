import argparse
import csv
import subprocess
import sys
from pathlib import Path
from datasets import load_dataset


def run_command(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def get_audio_stream_url(youtube_url: str) -> str:
    cmd = ["yt-dlp", "-f", "bestaudio", "--no-playlist", "--get-url", youtube_url]
    output = run_command(cmd)
    stream_url = output.splitlines()[0].strip() if output else ""
    return stream_url


def download_audio(stream_url: str, out_path: Path, start_s: float, duration_s: float) -> None:
    cmd = ["ffmpeg", "-y", "-ss", start_s, "-t", str(duration_s), "-i", stream_url,
           "-c:a", "pcm_s16le", "-ar", "32000", "-ac", "1", str(out_path)]
    run_command(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    # args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset('google/MusicCaps')

    dataset_csv = args.output_dir / "dataset.csv"
    wav_dir = args.output_dir / "wav"

    wav_dir.mkdir(parents=True, exist_ok=True)


    total = len(dataset['train'])
    success = 0
    skipped = 0
    failed = 0
    csv_rows: list[tuple[str, str]] = []

    for idx, row in enumerate(dataset['train']):
        try:
            ytid = row['ytid']
            start_s = row['start_s']
            end_s = row['end_s']
            caption = row['caption']

            youtube_url = f"https://www.youtube.com/watch?v={ytid}"
            out_name = f"{ytid}_{start_s}_{end_s}.wav"
            out_path = wav_dir / out_name

            if args.skip_existing and out_path.exists():
                skipped += 1
                csv_rows.append((out_name, caption))
                print(f"[{idx + 1}/{total}] SKIP exists: {out_path.name}")
                continue

            stream_url = get_audio_stream_url(youtube_url)
            download_audio(stream_url, out_path, start_s, end_s)
            success += 1

            csv_rows.append((out_name, caption))

            print(f"[{idx + 1}/{total}] OK   {out_path.name}")

        except Exception as exc:
            failed += 1
            print(f"[{idx + 1}/{total}] FAIL {exc}", file=sys.stderr)

    with dataset_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "caption"])
        writer.writerows(csv_rows)

    print("Done.")
    print(f"Downloaded: {success}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"CSV: {dataset_csv}")


if __name__ == "__main__":
    main()
