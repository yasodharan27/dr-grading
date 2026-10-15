"""Puts every model file listed in weights_manifest.json into the configured weights_dir.

    python scripts/download_weights.py                    # download (the release files and the two public ones)
    python scripts/download_weights.py --from-local DIR   # copy files with the release names from DIR instead
    python scripts/download_weights.py --inference-only   # skip the files needed only for training

Files already present with the expected size are kept. A file is checked against its listed size after download.
A download that times out or drops is retried up to 6 times, resuming from the bytes already on disk (HTTP Range),
with waits of 5, 10, 20, 40 and 60 seconds.
"""
import argparse
import json
import os
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drgrade.config import ROOT, load_config  # noqa: E402


ATTEMPTS = 6
WAITS = (5, 10, 20, 40, 60)                    # seconds before attempts 2 to 6
TIMEOUT = 120
CHUNK = 1 << 20


class DownloadError(RuntimeError):
    pass


def _open(url, start):
    headers = {"User-Agent": "drgrade-download"}
    if start:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT)


def fetch(url, target, expected_bytes, attempts=ATTEMPTS, waits=WAITS, sleep=time.sleep, log=print):
    """Downloads `url` to `target` with resume: after a timeout or a dropped connection the next attempt asks for the
    bytes still missing (HTTP Range) and appends them to the .part file. A server that ignores Range (answers 200)
    gets a full download again. The .part file is renamed to `target` only when its size equals `expected_bytes`."""
    part = target + ".part"
    last = None
    for attempt in range(1, attempts + 1):
        start = os.path.getsize(part) if os.path.exists(part) else 0
        if start >= expected_bytes:
            os.replace(part, target)
            return
        try:
            with _open(url, start) as response:
                status = getattr(response, "status", 200)
                mode = "ab" if (start and status == 206) else "wb"
                if start and status != 206:
                    start = 0
                with open(part, mode) as fh:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        fh.write(block)
            if os.path.getsize(part) == expected_bytes:
                os.replace(part, target)
                return
            last = f"incomplete transfer: {os.path.getsize(part)} of {expected_bytes} bytes"
        except (TimeoutError, socket.timeout, urllib.error.URLError, ConnectionError, OSError) as error:
            last = f"{type(error).__name__}: {error}"
        bytes_so_far = os.path.getsize(part) if os.path.exists(part) else 0
        if attempt == attempts:
            break
        wait = waits[min(attempt - 1, len(waits) - 1)]
        log(f"retry       {os.path.basename(target)}: attempt {attempt + 1} of {attempts} in {wait} s "
            f"({bytes_so_far} of {expected_bytes} bytes so far; {last})")
        sleep(wait)
    raise DownloadError(f"{os.path.basename(target)}: giving up after {attempts} attempts ({last}); "
                        f"the partial file is kept, so running the script again resumes it")


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
            try:
                fetch(entry["url"], target, entry["bytes"])
            except DownloadError as error:
                print(f"FAILED      {entry['file']}: {error}")
                failed.append(entry["file"])
                continue
            how = "downloaded"
        elif template:
            try:
                fetch(template.format(file=entry["file"]), target, entry["bytes"])
            except DownloadError as error:
                print(f"FAILED      {entry['file']}: {error}")
                failed.append(entry["file"])
                continue
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
