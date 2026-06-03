"""
Xoa video trung lap (noi dung giong het) trong data/labeled.
Giu 1 file / hash (uu tien file co so thu tu nho nhat trong ten).
"""
import argparse
import hashlib
import re
from pathlib import Path


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def numeric_suffix(path: Path) -> int:
    m = re.search(r"_(\d+)\.mp4$", path.name, re.IGNORECASE)
    return int(m.group(1)) if m else 10**9


def dedupe_folder(label_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    videos = sorted(label_dir.glob("*.mp4"), key=numeric_suffix)
    groups: dict[str, list[Path]] = {}
    for v in videos:
        groups.setdefault(file_hash(v), []).append(v)

    removed = 0
    kept = 0
    for paths in groups.values():
        paths.sort(key=numeric_suffix)
        keep = paths[0]
        kept += 1
        for dup in paths[1:]:
            removed += 1
            if dry_run:
                print(f"  [DRY] xoa {dup.name} (giu {keep.name})")
            else:
                dup.unlink()
    return kept, removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Xoa clip video trung lap trong labeled")
    parser.add_argument("--labeled_dir", type=str, default="data/labeled")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    labeled_dir = Path(__file__).resolve().parents[2] / args.labeled_dir
    total_kept = total_removed = 0

    for label_dir in sorted(labeled_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        kept, removed = dedupe_folder(label_dir, dry_run=args.dry_run)
        total_kept += kept
        total_removed += removed
        print(f"{label_dir.name}: giu {kept} | xoa {removed}")

    print(f"\nTong: giu {total_kept} | xoa {total_removed}")


if __name__ == "__main__":
    main()
