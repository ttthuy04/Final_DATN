"""
Pipeline: dedupe -> split 5s -> rename -> extract skeleton (--overwrite)
Mac dinh bo qua nhan 'distracted'.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
PRE = ROOT / "src" / "preprocess"


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_sec", type=float, default=5.0)
    parser.add_argument("--labels", type=str, default="")
    parser.add_argument("--skip", action="store_true", help="Bo qua dedupe/split/rename, chi extract")
    args = parser.parse_args()

    label_args = ["--labels", args.labels] if args.labels else []
    skip = ["--skip_labels", "distracted"]

    if not args.skip:
        run([PY, str(PRE / "remove_duplicate_clips.py")])
        run([PY, str(PRE / "split_labeled_clips.py"), "--max_sec", str(args.max_sec), *skip, *label_args])
        run([PY, str(PRE / "rename_labeled.py")])

    run([PY, str(PRE / "extract_skeleton.py"), "--overwrite", *skip, *label_args])
    print("\n[OK] Pipeline hoan tat.")


if __name__ == "__main__":
    main()
