"""
Đổi tên clip trong data/labeled/<label>/ thành <label>_1.mp4, <label>_2.mp4, ...
Sắp xếp theo thời gian sửa file (cũ -> mới).
"""
import argparse
from pathlib import Path


def rename_label_folder(label_dir: Path, dry_run: bool = False) -> int:
    label = label_dir.name
    files = sorted(label_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not files:
        return 0

    temp_names = []
    for idx, src in enumerate(files, start=1):
        temp = label_dir / f"__tmp_rename_{idx}.mp4"
        final = label_dir / f"{label}_{idx}.mp4"
        temp_names.append((src, temp, final))

    if dry_run:
        for src, _, final in temp_names:
            print(f"[DRY] {src.name} -> {final.name}")
        return len(temp_names)

    for src, temp, _ in temp_names:
        src.rename(temp)
    for _, temp, final in temp_names:
        temp.rename(final)

    return len(temp_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename labeled clips to label_1..label_n")
    parser.add_argument(
        "--labeled_dir",
        type=str,
        default="data/labeled",
        help="Thu muc labeled",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chi in ra, khong doi ten that",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Chi doi ten 1 nhan (vd: normal)",
    )
    args = parser.parse_args()

    labeled_dir = Path(args.labeled_dir)
    if not labeled_dir.exists():
        raise FileNotFoundError(f"Khong tim thay: {labeled_dir}")

    total = 0
    for label_dir in sorted(labeled_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        if args.label and label_dir.name != args.label:
            continue
        count = rename_label_folder(label_dir, dry_run=args.dry_run)
        print(f"{label_dir.name}: {count} file")
        total += count

    print(f"\nTong: {total} file")


if __name__ == "__main__":
    main()
