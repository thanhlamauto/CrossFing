#!/usr/bin/env python3
# fix_kaggle_paths.py
"""
Fix absolute local paths inside train_info/valid_info txts
to Kaggle dataset paths.

Example local path:
  /home/.../data/fvc/preproc/finger001_inst1.png

Becomes:
  /kaggle/input/<dataset-name>/fvc/preproc/finger001_inst1.png
"""

import os
import glob
import argparse

def fix_one_txt(txt_path, kaggle_root, anchor="/fvc/"):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    # safety: need at least 3 lines
    if len(lines) < 3:
        return False

    def convert_path(p):
        p = p.strip()
        if anchor in p:
            # keep suffix after /fvc/
            suffix = p.split(anchor, 1)[1]   # e.g. "preproc/xxx.png"
            return os.path.join(kaggle_root, suffix)
        else:
            # if already relative like "preproc/xxx.png"
            if not p.startswith("/") and not p.startswith("preproc"):
                return os.path.join(kaggle_root, p)
            return os.path.join(kaggle_root, p) if p.startswith("preproc") else p

    # line 1 and 2 are image paths in your format
    lines[1] = convert_path(lines[1]) + "\n"
    lines[2] = convert_path(lines[2]) + "\n"

    with open(txt_path, "w") as f:
        f.writelines(lines)

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="Path to fvc folder (contains preproc/, train_info/, valid_info/).")
    ap.add_argument("--kaggle_root", type=str, required=True,
                    help="Kaggle absolute root, e.g. /kaggle/input/<dataset-name>/fvc")
    ap.add_argument("--anchor", type=str, default="/fvc/",
                    help="Anchor substring to cut local prefix. Default: /fvc/")
    args = ap.parse_args()

    train_dir = os.path.join(args.data_root, "train_info")
    valid_dir = os.path.join(args.data_root, "valid_info")

    txts = glob.glob(os.path.join(train_dir, "*.txt")) + \
           glob.glob(os.path.join(valid_dir, "*.txt"))

    if len(txts) == 0:
        print("No txt files found under train_info/valid_info.")
        return

    ok = 0
    for t in txts:
        if fix_one_txt(t, args.kaggle_root, anchor=args.anchor):
            ok += 1

    print(f"Fixed {ok}/{len(txts)} txt files.")
    print("Example kaggle_root used:", args.kaggle_root)


if __name__ == "__main__":
    main()
