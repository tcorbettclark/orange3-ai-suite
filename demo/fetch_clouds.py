"""Fetch a curated cloud-genus corpus from the WMO International Cloud Atlas.

The Atlas (WMO-No. 407, cloudatlas.wmo.int) is the definitive reference for
cloud classification; its image gallery is per-genus. The Atlas' own filename
tags distinguish ground-observation photographs from schematics, satellite and
aerial views; the fetcher keeps the ``_clouds_m`` photographs whose filename
also names the requested genus.

Classes: five cloud genera (one line per genus in GENERA; extend freely).
The Atlas has no "clear sky" gallery (it classifies clouds), so the demo
corpus deliberately contains none.

Storage: ``data/clouds/<genus>/`` is untracked. The images remain the
property of their photographers; they are downloaded for personal,
non-commercial teaching use under the Atlas copyright terms. Full images are
downscaled to <= 512 px on save (the kernel's working size is 128 px anyway).
Also written:

- data/clouds/CREDITS.md   per-image photographer (©), location, date,
                           Atlas image ID and source URL + the WMO-No. 407
                           attribution
- data/review/<genus>.jpg  contact sheets for the eyeball pass (delete any
                           offender from data/clouds/<genus>/ and re-run;
                           the script is idempotent — kept files are kept)

Idempotent and re-runnable. Needs the dev venv (cv2) + network; ~5 minutes.

    uv run python demo/fetch_clouds.py
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import cv2
import numpy as np

BASE = "https://cloudatlas.wmo.int"
UA = (
    "orange3-nn-suite demo fetcher (educational, non-commercial; "
    "https://github.com/tcorbettclark/orange3-nn-suite)"
)

# genus label -> gallery API code
GENERAS = {
    "ci": "H1",  # Cirrus
    "ac": "M1",  # Altocumulus
    "sc": "L1",  # Stratocumulus
    "cb": "L4",  # Cumulonimbus
    "cu": "L3",  # Cumulus
}
GENUS_TOKENS = {
    "ci": "cirrus",
    "ac": "altocumulus",
    "sc": "stratocumulus",
    "cb": "cumulonimbus",
    "cu": "cumulus",
}

TARGET_DIR = Path(__file__).resolve().parent.parent / "data" / "clouds"
REVIEW_DIR = Path(__file__).resolve().parent.parent / "data" / "review"
MAX_EDGE = 512
JPEG_QUALITY = 88


def _get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"unreachable: {url}")


def gallery_ids(genus_code: str) -> list[tuple[int, str]]:
    """Photograph IDs + filenames for one genus from the gallery API.

    Filename convention: {caption}_{meteor-class}_{size}.jpg — the meteor
    class tag ('clouds', 'photometeors', ...) is our curation signal.
    """
    params = urllib.parse.urlencode({"tag": "", "cldg": genus_code, "cldt": ""})
    text = _get(f"{BASE}/en/api/?&{params}").decode("utf-8")
    items = []
    for line in text.splitlines():
        parts = line.split(";;")
        if len(parts) >= 2 and parts[0].isdigit():
            items.append((int(parts[0]), parts[1]))
    return items


def viewer_record(viewer_id: int) -> dict | None:
    """Parse imgviewer-{id}.txt for the primary photograph's attributes.

    The first ``<li>`` in the viewer holds the primary image: its
    ``compressed/{ID}_main_...`` source, photographer, location, date and
    title.
    """
    try:
        text = _get(f"{BASE}/en/imgviewer-{viewer_id}.txt").decode("utf-8")
    except RuntimeError:
        return None
    src_match = re.search(r"src=['\"]\.\./images/compressed/([^'\"]+)['\"]", text)
    if not src_match:
        return None
    filename = src_match.group(1).split("/")[-1]

    taken_match = re.search(r'\sid="takenby1"[^>]*>([^<]+)<', text)
    loc_match = re.search(r'\sid="loc1"[^>]*>([^<]+)<', text)
    date_match = re.search(r'\sid="photodate1"[^>]*>([^<]+)<', text)
    genus_match = re.search(r'<h1 class="img_title">([^<]+)<', text)
    return {
        "filename": filename,
        "photographer": (taken_match.group(1).strip() if taken_match else "")
        .replace("©", "")
        .replace("&#169;", "")
        .strip(),
        "location": loc_match.group(1).strip() if loc_match else "",
        "photodate": date_match.group(1).strip() if date_match else "",
        "img_title": genus_match.group(1).strip() if genus_match else "",
        "source": f"{BASE}/en/imgviewer-{viewer_id}.txt",
    }


def image_url(rec: dict) -> str:
    # e.g. ../images/compressed/2635_main_cumulus-humilis_clouds.jpg
    return f"{BASE}/images/compressed/{rec['filename']}"


def save_downscaled(raw: bytes, dest: Path) -> bool:
    """Downscale to <= MAX_EDGE on the long edge; reject undecodable files."""
    array = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        return False
    h, w = array.shape[:2]
    scale = min(1.0, MAX_EDGE / max(h, w))
    if scale < 1.0:
        array = cv2.resize(array, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.imwrite(str(dest), array, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def contact_sheet(paths: list[Path], dest: Path) -> None:
    cols, cell = 10, 180
    rows = max(1, (len(paths) + cols - 1) // cols)
    sheet = np.full((rows * cell, cols * cell, 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, (cell - 10, cell - 26), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y, x = r * cell, c * cell
        sheet[y + 22 : y + cell - 4, x + 5 : x + cell - 5] = img
        cv2.putText(sheet, f"#{i}", (x + 8, y + 16), font, 0.45, (0, 0, 200), 1)
    cv2.imwrite(str(dest), sheet)


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = TARGET_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for genus, code in GENERAS.items():
        out_dir = TARGET_DIR / genus
        out_dir.mkdir(parents=True, exist_ok=True)
        entries = manifest.setdefault(genus, {})

        existing = sorted(out_dir.glob("*.jpg"))
        if existing:
            print(
                f"{genus}: {len(existing)} images already fetched — skipping "
                "(delete files or manifest entries to refetch)"
            )
            continue

        kept: list[Path] = []
        for viewer_id, api_filename in gallery_ids(code):
            # curation by the Atlas' own filename tags: real cloud photographs
            # are tagged '_clouds'; photometeors/lithometeors have their own
            # tags. The genus name in the filename guards cross-genus leaks
            # (a photo listed for its *associated* genus). Cumulonimbus is
            # safe: "cumulus" is not a substring of "cumulonimbus".
            if not api_filename.endswith(("_clouds_m.jpg", "_clouds_m.JPG")):
                continue
            if GENUS_TOKENS[genus] not in api_filename.lower():
                continue
            rec = viewer_record(viewer_id)
            if rec is None:
                continue
            fname = (
                re.sub(r"[^A-Za-z0-9]+", "_", rec["img_title"])[:32].strip("_")
                + f"_{viewer_id}.jpg"
            )
            if fname in entries:
                continue
            try:
                raw = _get(image_url(rec))
            except RuntimeError:
                continue
            dest = out_dir / fname
            if not save_downscaled(raw, dest):
                continue
            entries[fname] = rec
            kept.append(dest)
            time.sleep(0.1)
        print(
            f"{genus}: kept {len(kept)} ground-observation photographs (from gallery code {code})"
        )

        if kept:
            contact_sheet(sorted(kept), REVIEW_DIR / f"{genus}.jpg")

    manifest_path.write_text(json.dumps(manifest, indent=1))

    counts = {
        c.name: len(list(c.glob("*.jpg"))) for c in sorted(TARGET_DIR.iterdir()) if c.is_dir()
    }
    credits = [
        "# Image credits — WMO International Cloud Atlas",
        "",
        f"Generated {date.today():%Y-%m-%d} by `demo/fetch_clouds.py` from the",
        "[WMO International Cloud Atlas](https://cloudatlas.wmo.int/) (WMO-No. 407),",
        "the World Meteorological Organization's reference standard for cloud",
        "classification. Images are © their photographers as noted below and",
        "remain so; they were obtained here for *personal, non-commercial use*",
        "per the Atlas copyright terms, and are **not** covered by the",
        "repository's MIT code licence.",
        "",
    ]
    for genus, items in sorted(manifest.items()):
        credits += [f"## {genus} ({len(items)} images)", ""]
        for fname, rec in sorted(items.items()):
            photographer = (
                (rec.get("photographer") or "WMO").replace("&#169;", "").replace("©", "").strip()
            )
            bits = [f"© {photographer or 'WMO'}"]
            if rec.get("location"):
                bits.append(str(rec["location"]))
            if rec.get("photodate"):
                bits.append(f"taken {rec['photodate']}")
            bits.append(
                f"Atlas image ID {rec['source'].rsplit('-', 1)[-1].split('.')[0]}"
                if rec.get("source")
                else ""
            )
            credits.append(
                f"- `{fname}` ({rec.get('img_title', '')}) — "
                + " — ".join(b for b in bits if b)
                + f" — {rec.get('source', '')}"
            )
        credits.append("")
    (TARGET_DIR / "CREDITS.md").write_text("\n".join(credits))

    print("final counts:", counts)
    print(f"review contact sheets in {REVIEW_DIR}")


if __name__ == "__main__":
    main()
