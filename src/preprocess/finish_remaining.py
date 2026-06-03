"""Sau khi split xong: rename + extract skeleton cho sleep, standing, use_phone."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
PRE = ROOT / "src" / "preprocess"
LABELS = ["sleep", "standing", "use_phone"]


def count_unsplit(label: str) -> int:
    d = ROOT / "data" / "labeled" / label
    return sum(1 for p in d.glob("*.mp4") if "_seg" not in p.stem)


def main() -> None:
    for label in LABELS:
        n = count_unsplit(label)
        if n > 0:
            print(f"[WAIT] {label} con {n} video chua tach. Chay split truoc.")
            return

    for label in LABELS:
        subprocess.run([PY, str(PRE / "rename_labeled.py"), "--label", label], cwd=ROOT, check=True)
        subprocess.run(
            [PY, str(PRE / "extract_skeleton.py"), "--overwrite", "--labels", label],
            cwd=ROOT,
            check=True,
        )
    print("[OK] Hoan tat rename + extract cho sleep, standing, use_phone")


if __name__ == "__main__":
    main()
