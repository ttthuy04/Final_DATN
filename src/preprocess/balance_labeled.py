"""
Cân bằng số clip mỗi lớp trong data/labeled/<label>/.
- Thừa: giữ 200 file đầu (theo số thứ tự trong tên), xóa phần còn lại.
- Thiếu: nhân bản vòng từ các file hiện có cho đủ 200.
Sau đó gọi rename để đặt lại label_1..label_200.
"""
import argparse
import re
import shutil
from pathlib import Path


TARGET_PER_CLASS = 200


def numeric_key(path: Path) -> int:
    match = re.search(r"_(\d+)\.mp4$", path.name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 10**9


def balance_folder(label_dir: Path, target: int = TARGET_PER_CLASS) -> tuple[int, int, int, int]:
    label = label_dir.name
    files = sorted(label_dir.glob("*.mp4"), key=numeric_key)
    before = len(files)

    if before == 0:
        return before, 0, 0, 0

    deleted = 0
    copied = 0

    if before > target:
        for extra in files[target:]:
            extra.unlink()
            deleted += 1
        files = files[:target]

    if before < target:
        if not files:
            return before, deleted, copied

        source_files = list(files)
        idx = 0
        next_index = max((numeric_key(p) for p in source_files), default=0) + 1
        while len(files) < target:
            src = source_files[idx % len(source_files)]
            dst = label_dir / f"{label}_{next_index}.mp4"
            while dst.exists():
                next_index += 1
                dst = label_dir / f"{label}_{next_index}.mp4"
            shutil.copy2(src, dst)
            files.append(dst)
            copied += 1
            next_index += 1
            idx += 1

    after = len(list(label_dir.glob("*.mp4")))
    return before, deleted, copied if before < target else 0, after


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance labeled clips to fixed count per class")
    parser.add_argument("--labeled_dir", type=str, default="data/labeled")
    parser.add_argument("--target", type=int, default=TARGET_PER_CLASS)
    parser.add_argument("--no-rename", action="store_true", help="Khong doi ten lai sau khi can bang")
    args = parser.parse_args()

    labeled_dir = Path(args.labeled_dir)
    if not labeled_dir.exists():
        raise FileNotFoundError(f"Khong tim thay: {labeled_dir}")

    print(f"Muc tieu: {args.target} clip / lop\n")

    for label_dir in sorted(labeled_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        before, deleted, copied, after = balance_folder(label_dir, args.target)
        action = []
        if deleted:
            action.append(f"xoa {deleted}")
        if copied:
            action.append(f"copy {copied}")
        action_str = ", ".join(action) if action else "giu nguyen"
        print(f"{label_dir.name}: {before} -> {after} ({action_str})")

    if not args.no_rename:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from rename_labeled import rename_label_folder

        print("\nDoi ten lai theo thu tu 1..n:")
        for label_dir in sorted(labeled_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            n = rename_label_folder(label_dir, dry_run=False)
            print(f"  {label_dir.name}: {n} file")


if __name__ == "__main__":
    main()
