#!/usr/bin/env python3
"""Build LoRA training dataset from Wikimedia Commons with multi-stage filtering.

Usage:
    python scripts/build_lora_dataset.py
    python scripts/build_lora_dataset.py --from-metadata docs/lora_dataset_metadata.csv
"""

import argparse
from collections import defaultdict
import csv
import html
import os
from pathlib import Path
import random
import re
import time
import warnings
from typing import Any

import cv2
import imagehash
import numpy as np
from PIL import Image
import requests
import yaml

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

# Lazy model references
_YOLO_MODEL = None
_YOLO_TRIED = False
_USE_HOG = False
_HOG_WARNING_PRINTED = False


def get_person_detector():
    """Lazily load YOLOv8n or fallback to OpenCV HOG."""
    global _YOLO_MODEL, _YOLO_TRIED, _USE_HOG, _HOG_WARNING_PRINTED
    if not _YOLO_TRIED:
        _YOLO_TRIED = True
        try:
            from ultralytics import YOLO
            _YOLO_MODEL = YOLO("yolov8n.pt")
        except Exception as e:
            if not _HOG_WARNING_PRINTED:
                print(f"WARNING: Could not load ultralytics YOLO ({e}). Falling back to OpenCV HOG detector.")
                _HOG_WARNING_PRINTED = True
            _USE_HOG = True
    return _YOLO_MODEL, _USE_HOG


def check_person_yolo(
    img_bgr: np.ndarray,
    model,
    conf_thresh: float = 0.35,
    min_area_fraction: float = 0.015,
) -> bool:
    """Return True if a person is detected with conf >= conf_thresh and area >= min_area_fraction."""
    img_h, img_w = img_bgr.shape[:2]
    img_area = img_h * img_w
    if img_area <= 0:
        return False

    results = model(img_bgr, verbose=False)
    if not results or len(results) == 0:
        return False

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return False

    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        cls_name = model.names.get(cls_id, "") if hasattr(model, "names") else ""
        if (cls_id == 0 or cls_name == "person") and conf >= conf_thresh:
            xyxy = box.xyxy[0].tolist()
            w = max(0.0, float(xyxy[2] - xyxy[0]))
            h = max(0.0, float(xyxy[3] - xyxy[1]))
            box_area = w * h
            if (box_area / img_area) >= min_area_fraction:
                return True
    return False


def check_person_hog(img_bgr: np.ndarray, min_area_fraction: float = 0.015) -> bool:
    """Fallback person detector using OpenCV HOG."""
    img_h, img_w = img_bgr.shape[:2]
    img_area = img_h * img_w
    if img_area <= 0:
        return False

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(img_bgr, winStride=(8, 8), padding=(4, 4), scale=1.05)
    for (x, y, w, h), weight in zip(rects, weights):
        w_val = float(weight[0]) if hasattr(weight, "__iter__") else float(weight)
        box_area = float(w * h)
        if (box_area / img_area) >= min_area_fraction and w_val >= 0.35:
            return True
    return False


def detect_person(
    img_bgr: np.ndarray,
    model=None,
    use_hog: bool = False,
    conf_thresh: float = 0.35,
    min_area_fraction: float = 0.015,
) -> bool:
    """Detect person using YOLOv8n or OpenCV HOG fallback."""
    if model is not None and not use_hog:
        return check_person_yolo(img_bgr, model, conf_thresh, min_area_fraction)
    if use_hog:
        return check_person_hog(img_bgr, min_area_fraction)

    m, hog = get_person_detector()
    if hog or m is None:
        return check_person_hog(img_bgr, min_area_fraction)
    return check_person_yolo(img_bgr, m, conf_thresh, min_area_fraction)


def check_dark_or_low_res(
    img_bgr: np.ndarray,
    min_luminance: float = 40.0,
    min_short_side: int = 600,
) -> bool:
    """Return True if image is too dark (mean luminance < 40) or low resolution (short side < 600)."""
    h, w = img_bgr.shape[:2]
    if min(h, w) < min_short_side:
        return True
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))
    if mean_lum < min_luminance:
        return True
    return False


def compute_phash(img_bgr: np.ndarray) -> imagehash.ImageHash:
    """Compute perceptual hash of image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    return imagehash.phash(pil_img)


def is_duplicate(phash: imagehash.ImageHash, existing_hashes: list[imagehash.ImageHash], max_distance: int = 6) -> bool:
    """Return True if phash Hamming distance <= max_distance to any existing hash."""
    for existing in existing_hashes:
        if (phash - existing) <= max_distance:
            return True
    return False


def clean_author(raw_author: str | None) -> str:
    """Clean and unescape author string, stripping HTML tags."""
    if not raw_author:
        return "Unknown"
    cleaned = re.sub(r"<[^>]+>", "", str(raw_author))
    cleaned = html.unescape(cleaned).strip()
    return cleaned if cleaned else "Unknown"


def clean_license(raw_license: str | None) -> str:
    """Clean and unescape license string, stripping HTML tags."""
    if not raw_license:
        return "Unknown"
    cleaned = re.sub(r"<[^>]+>", "", str(raw_license))
    cleaned = html.unescape(cleaned).strip()
    return cleaned if cleaned else "Unknown"


def process_and_crop(img_bgr: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Resize shorter side to target_size using INTER_AREA, then center crop to target_size x target_size."""
    h, w = img_bgr.shape[:2]
    if h < w:
        new_h = target_size
        new_w = int(round(w * (target_size / h)))
    else:
        new_w = target_size
        new_h = int(round(h * (target_size / w)))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    start_y = max(0, (new_h - target_size) // 2)
    start_x = max(0, (new_w - target_size) // 2)
    crop = resized[start_y:start_y + target_size, start_x:start_x + target_size]
    return crop


def build_caption(trigger: str, caption_phrase: str, theme_description: str) -> str:
    """Format caption string: '<trigger>, <term caption phrase>, <theme description>'."""
    return f"{trigger}, {caption_phrase}, {theme_description}"


def save_metadata_csv(records: list[dict[str, Any]], path: Path | str) -> None:
    """Save dataset metadata to CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "theme",
        "term",
        "caption",
        "source_url",
        "page_url",
        "author",
        "license",
        "phash",
    ]
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def load_metadata_csv(path: Path | str) -> list[dict[str, str]]:
    """Load dataset metadata from CSV."""
    p = Path(path)
    records = []
    with open(p, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(dict(r))
    return records


def generate_overview_sheet(
    dataset_dir: Path | str,
    output_path: Path | str,
    themes: list[str],
    seed: int = 42,
    thumb_size: int = 160,
    label_width: int = 160,
) -> None:
    """Generate a contact sheet with 1 row per theme, 12 random thumbnails per row."""
    dataset_path = Path(dataset_dir)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    rows = []
    for theme in themes:
        theme_dir = dataset_path / theme
        img_files = sorted(list(theme_dir.glob("*.jpg")))
        if not img_files:
            continue

        if len(img_files) >= 12:
            chosen_files = rng.sample(img_files, 12)
        else:
            chosen_files = img_files

        # Label tile
        label_tile = np.full((thumb_size, label_width, 3), 32, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        text = theme.replace("_", " ")
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        tx = max(8, (label_width - tw) // 2)
        ty = (thumb_size + th) // 2
        cv2.putText(label_tile, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        row_tiles = [label_tile]
        for f in chosen_files:
            img = cv2.imread(str(f))
            if img is not None:
                thumb = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
            else:
                thumb = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
            row_tiles.append(thumb)

        # Pad to exactly 12 thumbnails if theme has fewer
        while len(row_tiles) < 13:
            row_tiles.append(np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8))

        row_img = np.hstack(row_tiles)
        rows.append(row_img)

    if rows:
        contact_sheet = np.vstack(rows)
        cv2.imwrite(str(out_path), contact_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def request_with_backoff(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
) -> requests.Response | None:
    """Execute HTTP GET with polite delay and exponential backoff on 429/503."""
    for attempt in range(max_retries):
        time.sleep(1.0)
        try:
            r = session.get(url, params=params, timeout=25)
            if r.status_code in (429, 503):
                retry_after = r.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after and retry_after.isdigit() else initial_backoff * (2 ** attempt)
                time.sleep(wait_time)
                continue
            if r.status_code == 200:
                return r
        except Exception:
            time.sleep(initial_backoff * (2 ** attempt))
    return None


def fetch_candidate_batch(
    session: requests.Session,
    search_term: str,
    continue_params: dict[str, Any] | None = None,
    max_retries: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Fetch one batch of up to 50 search candidates from Wikimedia Commons."""
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrlimit": 50,
        "gsrsearch": search_term,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1280,
        "format": "json",
    }
    if continue_params:
        params.update(continue_params)

    data = None
    for attempt in range(max_retries):
        resp = request_with_backoff(session, api_url, params=params)
        if resp is None:
            time.sleep(2.0 * (2 ** attempt))
            continue
        try:
            data = resp.json()
            break
        except Exception:
            time.sleep(2.0 * (2 ** attempt))

    if data is None:
        return [], None

    pages = data.get("query", {}).get("pages", {})
    sorted_pages = sorted(pages.values(), key=lambda p: p.get("index", 999999))

    candidates = []
    for page in sorted_pages:
        imageinfo = page.get("imageinfo", [])
        if not imageinfo:
            continue
        ii = imageinfo[0]
        mime = ii.get("mime", "")
        if mime not in ("image/jpeg", "image/png", "image/webp"):
            continue

        raw_artist = ii.get("extmetadata", {}).get("Artist", {}).get("value")
        raw_license = (
            ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value")
            or ii.get("extmetadata", {}).get("License", {}).get("value")
            or ii.get("extmetadata", {}).get("UsageTerms", {}).get("value")
        )

        candidates.append({
            "title": page.get("title", ""),
            "page_url": ii.get("descriptionurl", ""),
            "source_url": ii.get("thumburl") or ii.get("url", ""),
            "download_url": ii.get("thumburl") or ii.get("url", ""),
            "width": int(ii.get("width", 0)),
            "height": int(ii.get("height", 0)),
            "author": clean_author(raw_artist),
            "license": clean_license(raw_license),
        })

    next_continue = data.get("continue")
    return candidates, next_continue


def to_thumb_url(url: str, width: int = 1280) -> str:
    """Convert Wikimedia Commons upload URL to 1280px thumbnail on thumb.wikimedia.org."""
    if not url:
        return ""
    # Always ensure host is thumb.wikimedia.org to prevent 429 rate limiting on upload.wikimedia.org
    url = url.replace("upload.wikimedia.org", "thumb.wikimedia.org")
    if "/thumb/" in url:
        return url
    m = re.match(r"https?://(?:upload|thumb)\.wikimedia\.org/wikipedia/commons/([0-9a-fA-F]/[0-9a-fA-F]{2})/([^/?#]+)", url)
    if m:
        path_hash = m.group(1)
        filename = m.group(2)
        return f"https://thumb.wikimedia.org/wikipedia/commons/thumb/{path_hash}/{filename}/{width}px-{filename}"
    return url


def download_candidate_image(session: requests.Session, url: str, timeout: float = 12.0) -> np.ndarray | None:
    """Download candidate image via thumb cache with 1s etiquette sleep and no slow backoff retries."""
    time.sleep(1.0)
    thumb_url = to_thumb_url(url)
    try:
        r = session.get(thumb_url, timeout=timeout)
        if r.status_code == 200 and r.content:
            arr = np.frombuffer(r.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
    except Exception:
        pass
    return None


def run_repro_mode(metadata_csv: Path | str, base_out_dir: Path | str = "data/lora_train") -> None:
    """Rebuild exact dataset from metadata CSV without filtering."""
    metadata_path = Path(metadata_csv)
    records = load_metadata_csv(metadata_path)
    session = requests.Session()
    session.headers.update({"User-Agent": "indic-product-photo-studio course project, student use"})

    print(f"Repro mode: rebuilding dataset from {metadata_path} ({len(records)} images)...")
    for row in records:
        theme = row["theme"]
        filename = row["filename"]
        source_url = row["source_url"]
        caption = row["caption"]

        theme_dir = Path(base_out_dir) / theme
        theme_dir.mkdir(parents=True, exist_ok=True)
        img_out = theme_dir / filename
        txt_out = theme_dir / (Path(filename).stem + ".txt")

        download_url = to_thumb_url(source_url)
        resp = request_with_backoff(session, download_url)
        if resp is None or resp.status_code != 200:
            print(f"Warning: Failed to download {download_url}")
            continue

        arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Failed to decode image from {source_url}")
            continue

        crop = process_and_crop(img, target_size=512)
        cv2.imwrite(str(img_out), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        txt_out.write_text(caption, encoding="utf-8")

    print("Repro mode complete.")


def main():
    parser = argparse.ArgumentParser(description="LoRA training dataset builder from Wikimedia Commons")
    parser.add_argument("--config", default="configs/lora_themes.yaml", help="Path to themes YAML config")
    parser.add_argument("--metadata-out", default="docs/lora_dataset_metadata.csv", help="Path to metadata CSV output")
    parser.add_argument("--overview-out", default="outputs/lora_dataset_overview.jpg", help="Path to overview image")
    parser.add_argument("--data-dir", default="data/lora_train", help="Base dataset output directory")
    parser.add_argument("--from-metadata", help="Rebuild dataset from metadata CSV without filtering")
    parser.add_argument("--target-per-theme", type=int, default=60, help="Target images per theme")
    args = parser.parse_args()

    if args.from_metadata:
        run_repro_mode(args.from_metadata, base_out_dir=args.data_dir)
        return

    with open(args.config, "r", encoding="utf-8") as f:
        themes_cfg = yaml.safe_load(f)

    session = requests.Session()
    session.headers.update({"User-Agent": "indic-product-photo-studio course project, student use"})

    # Ensure detector is ready
    get_person_detector()

    base_data_path = Path(args.data_dir)
    base_data_path.mkdir(parents=True, exist_ok=True)

    all_accepted_hashes: list[imagehash.ImageHash] = []
    all_metadata_records: list[dict[str, Any]] = []

    stats: dict[str, dict[str, int]] = {}

    for theme, theme_info in themes_cfg.items():
        trigger = theme_info["trigger"]
        description = theme_info["description"]
        main_terms = theme_info.get("terms", [])
        fallback_terms = theme_info.get("fallback_terms", [])

        theme_dir = base_data_path / theme
        theme_dir.mkdir(parents=True, exist_ok=True)

        theme_stats = {
            "candidates_found": 0,
            "dropped_person": 0,
            "dropped_dark_small": 0,
            "dropped_duplicate": 0,
            "dropped_author_cap": 0,
            "final_count": 0,
        }
        theme_author_counts: dict[str, int] = defaultdict(int)
        theme_accepted_count = 0

        print(f"\nProcessing theme: {theme} (trigger: '{trigger}')")

        def evaluate_term(term_dict: dict[str, str], is_fallback: bool = False):
            nonlocal theme_accepted_count
            search_query = term_dict["term"]
            caption_phrase = term_dict["caption"]
            tag = "[fallback]" if is_fallback else "[main]"
            print(f"  {tag} Query: '{search_query}' (phrase: '{caption_phrase}')")

            continue_token = None
            pages_to_fetch = 5  # can fetch up to 5 pages (250 candidates) per term if needed

            for _ in range(pages_to_fetch):
                if theme_accepted_count >= args.target_per_theme:
                    break

                candidates, continue_token = fetch_candidate_batch(session, search_query, continue_token)
                if not candidates:
                    break

                for cand in candidates:
                    if theme_accepted_count >= args.target_per_theme:
                        break

                    theme_stats["candidates_found"] += 1

                    # Download candidate image
                    img = download_candidate_image(session, cand["download_url"])
                    if img is None:
                        continue

                    # FILTER 1: Person detection
                    if detect_person(img):
                        theme_stats["dropped_person"] += 1
                        continue

                    # FILTER 2: Dark or low-res (luminance < 40 or short side < 600)
                    if check_dark_or_low_res(img, min_luminance=40.0, min_short_side=600):
                        theme_stats["dropped_dark_small"] += 1
                        continue

                    # FILTER 3: Near-duplicates (phash dist <= 6 to ANY accepted across all themes)
                    phash = compute_phash(img)
                    if is_duplicate(phash, all_accepted_hashes, max_distance=6):
                        theme_stats["dropped_duplicate"] += 1
                        continue

                    # FILTER 4: Author diversity (max 6 images per author per theme)
                    author = cand["author"]
                    if theme_author_counts[author] >= 6:
                        theme_stats["dropped_author_cap"] += 1
                        continue

                    # Image accepted!
                    theme_author_counts[author] += 1
                    all_accepted_hashes.append(phash)
                    theme_accepted_count += 1
                    theme_stats["final_count"] = theme_accepted_count
                    print(f"    [{theme}] Accepted {theme_accepted_count}/{args.target_per_theme}: {cand['title']}", flush=True)

                    # Process: resize shorter side to 512, center crop 512x512
                    crop = process_and_crop(img, target_size=512)
                    img_filename = f"{theme_accepted_count:03d}.jpg"
                    txt_filename = f"{theme_accepted_count:03d}.txt"

                    out_img_path = theme_dir / img_filename
                    out_txt_path = theme_dir / txt_filename

                    cv2.imwrite(str(out_img_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                    caption_text = build_caption(trigger, caption_phrase, description)
                    out_txt_path.write_text(caption_text, encoding="utf-8")

                    all_metadata_records.append({
                        "filename": img_filename,
                        "theme": theme,
                        "term": search_query,
                        "caption": caption_text,
                        "source_url": cand["source_url"],
                        "page_url": cand["page_url"],
                        "author": author,
                        "license": cand["license"],
                        "phash": str(phash),
                    })

                if not continue_token:
                    break

        # Run main terms
        for term_dict in main_terms:
            if theme_accepted_count >= args.target_per_theme:
                break
            evaluate_term(term_dict, is_fallback=False)

        # If theme < 50 and hasn't reached target, run fallback terms
        if theme_accepted_count < 50 and theme_accepted_count < args.target_per_theme:
            print(f"  Theme '{theme}' has {theme_accepted_count} (< 50) images. Running fallback terms...")
            for fb_dict in fallback_terms:
                if theme_accepted_count >= args.target_per_theme:
                    break
                evaluate_term(fb_dict, is_fallback=True)

        stats[theme] = theme_stats
        print(f"  Theme '{theme}' finished with {theme_accepted_count} accepted images.")

    # Save outputs
    save_metadata_csv(all_metadata_records, args.metadata_out)
    print(f"\nMetadata saved to {args.metadata_out} ({len(all_metadata_records)} rows)")

    generate_overview_sheet(
        dataset_dir=args.data_dir,
        output_path=args.overview_out,
        themes=list(themes_cfg.keys()),
        seed=42,
    )
    print(f"Overview contact sheet saved to {args.overview_out}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Theme':<16} {'Candidates':<12} {'Person Drop':<14} {'Dark/Small':<12} {'Duplicates':<12} {'Author Cap':<12} {'Final Count':<12}")
    print("-" * 90)
    for theme, s in stats.items():
        print(
            f"{theme:<16} {s['candidates_found']:<12} {s['dropped_person']:<14} "
            f"{s['dropped_dark_small']:<12} {s['dropped_duplicate']:<12} {s['dropped_author_cap']:<12} "
            f"{s['final_count']:<12}"
        )
    print("=" * 90)

    # Warnings for any theme < 40
    for theme, s in stats.items():
        if s["final_count"] < 40:
            print(f"WARNING: Theme '{theme}' has only {s['final_count']} images (< 40)!")


if __name__ == "__main__":
    main()
