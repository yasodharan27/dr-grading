"""Puts every model file listed in weights_manifest.json into the configured weights_dir.

    python scripts/download_weights.py                    # download (the release files and the two public ones)
    python scripts/download_weights.py --from-local DIR   # copy files with the release names from DIR instead
    python scripts/download_weights.py --inference-only   # skip the files needed only for training

Files already present with the expected size are kept. A file is checked against its listed size after download.
"""
import argparse
import json
import os
import shutil
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drgrade.config import ROOT, load_config  # noqa: E402


def fetch(url, target):
    tmp = target + ".part"
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as fh:
        shutil.copyfileobj(response, fh, length=1 << 20)
    os.replace(tmp, target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--from-local", help="folder holding the release files under their release names")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--only", nargs="+", metavar="FILE", help="download only these files of the manifest")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    with open(os.path.join(ROOT, "weights_manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    os.makedirs(cfg["weights_dir"], exist_ok=True)
    release = manifest["release"]
    template = release.get("url_template") or (release["base_url"].rstrip("/") + "/{file}" if release.get("base_url") else None)
    failed = []
    for entry in manifest["files"]:
        if args.inference_only and entry.get("training_only"):
            continue
        if args.only and entry["file"] not in args.only:
            continue
        target = os.path.join(cfg["weights_dir"], entry["file"])
        if os.path.exists(target) and os.path.getsize(target) == entry["bytes"]:
            print(f"present     {entry['file']}")
            continue
        local = os.path.join(args.from_local, entry["file"]) if args.from_local else None
        if local and os.path.exists(local):
            shutil.copyfile(local, target)
            how = "copied"
        elif entry["source"] == "url":
            fetch(entry["url"], target)
            how = "downloaded"
        elif template:
            fetch(template.format(file=entry["file"]), target)
            how = "downloaded"
        else:
            print(f"UNAVAILABLE {entry['file']}: no release location is set in weights_manifest.json; use --from-local")
            failed.append(entry["file"])
            continue
        size = os.path.getsize(target)
        if size != entry["bytes"]:
            os.remove(target)
            print(f"FAILED      {entry['file']}: {size} bytes, expected {entry['bytes']}")
            failed.append(entry["file"])
        else:
            print(f"{how:<11} {entry['file']}")
    if failed:
        sys.exit(f"{len(failed)} file(s) missing: {failed}")
    print(f"all weights in {cfg['weights_dir']}")


if __name__ == "__main__":
    main()
