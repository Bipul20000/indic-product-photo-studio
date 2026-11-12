#!/usr/bin/env python3
"""Build LoRA training dataset from Wikimedia Commons with CLIP relevance filtering.

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
import shutil
import time
from typing import Any

import cv2
import imagehash
import numpy as np
from PIL import Image
import requests
import yaml

from studio.vision.people import contains_person


def check_person_yolo(img_bgr: np.ndarray, model: Any = None) -> bool:
    """Helper for mocked person detector verification with Phase 4 filtering thresholds."""
    return contains_person(img_bgr, conf=0.35, min_box_area=0.015, ignore_mask=None, model=model)


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
    crop = resized[start_y : start_y + target_size, start_x : start_x + target_size]
    return crop


def build_caption(trigger: str, caption_phrase: str, theme_description: str) -> str:
    """Format caption string: '<trigger>, <caption phrase>, <theme description>'."""
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
        "clip_p_pos",
        "clip_best_positive",
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


# Global lazy references for CLIP
_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def get_clip_model_and_processor():
    """Lazily load openai/clip-vit-base-patch32 model and processor globally."""
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model_id = "openai/clip-vit-base-patch32"
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_id)
        _CLIP_MODEL = CLIPModel.from_pretrained(model_id, use_safetensors=False).to("cpu", dtype=torch.float32)
        _CLIP_MODEL.eval()
    return _CLIP_MODEL, _CLIP_PROCESSOR


def compute_clip_probs(logits: np.ndarray, num_positives: int) -> tuple[float, int]:
    """Compute softmax over logits, summed positive probability, and best positive index.

    Args:
        logits: 1D array of shape (num_prompts,) containing unnormalized logit scores.
        num_positives: Number of positive prompts (the first num_positives entries).

    Returns:
        (p_pos, best_positive_index)
    """
    if logits.ndim > 1:
        logits = logits.squeeze()
    # Numerically stable softmax
    shifted = logits - np.max(logits)
    exp_l = np.exp(shifted)
    probs = exp_l / np.sum(exp_l)

    p_pos = float(np.sum(probs[:num_positives]))
    best_pos_idx = int(np.argmax(probs[:num_positives]))
    return p_pos, best_pos_idx


def filter_and_rank_by_clip(
    candidates: list[dict[str, Any]],
    min_count: int = 40,
    top_k: int = 60,
    thresholds: tuple[float, ...] = (0.6, 0.5, 0.4),
) -> tuple[list[dict[str, Any]], float, int]:
    """Filter images by CLIP positive probability with threshold relaxation and take top_k.

    Threshold sequence: 0.6 -> 0.5 -> 0.4 until >= min_count images pass.
    Then rank by p_pos descending and take top_k.

    Returns:
        (kept_candidates, final_threshold, passed_count_at_final_threshold)
    """
    final_thresh = thresholds[0]
    passed = []

    for thresh in thresholds:
        final_thresh = thresh
        passed = [c for c in candidates if c.get("clip_p_pos", 0.0) >= thresh]
        if len(passed) >= min_count:
            break

    if final_thresh < thresholds[0]:
        print(f"WARNING: Relaxed CLIP threshold to {final_thresh} (candidates passed: {len(passed)})")

    # Rank by clip_p_pos descending
    passed.sort(key=lambda x: x.get("clip_p_pos", 0.0), reverse=True)
    kept = passed[:top_k]
    return kept, final_thresh, len(passed)


def score_images_with_clip(
    images_bgr: list[np.ndarray],
    positive_prompts: list[dict[str, str]],
    negative_prompts: list[str],
    batch_size: int = 16,
) -> list[tuple[float, str, str]]:
    """Score a list of BGR images against positive and negative prompts in batches of 16 on CPU.

    Returns:
        List of (p_pos, best_positive_prompt, best_positive_caption)
    """
    if not images_bgr:
        return []

    import torch
    from PIL import Image

    model, processor = get_clip_model_and_processor()

    pos_texts = [p["prompt"] for p in positive_prompts]
    all_texts = pos_texts + list(negative_prompts)
    num_pos = len(positive_prompts)

    results: list[tuple[float, str, str]] = []
    total_imgs = len(images_bgr)

    for i in range(0, total_imgs, batch_size):
        batch_bgr = images_bgr[i : i + batch_size]
        batch_pil = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in batch_bgr]

        inputs = processor(text=all_texts, images=batch_pil, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            # Logits: (batch_size, num_texts)
            logits = outputs.logits_per_image.cpu().numpy()

        for row_logits in logits:
            p_pos, best_idx = compute_clip_probs(row_logits, num_pos)
            best_prompt = positive_prompts[best_idx]["prompt"]
            best_caption = positive_prompts[best_idx]["caption"]
            results.append((p_pos, best_prompt, best_caption))

    return results


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


def generate_overview_sheet_from_images(
    theme_images: dict[str, list[np.ndarray]],
    output_path: Path | str,
    themes: list[str],
    thumb_size: int = 160,
    label_width: int = 160,
) -> None:
    """Generate a contact sheet directly from image arrays with 1 row per theme, 12 thumbnails."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for theme in themes:
        img_list = theme_images.get(theme, [])
        if not img_list:
            continue

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
        for img in img_list[:12]:
            if img is not None:
                thumb = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
            else:
                thumb = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
            row_tiles.append(thumb)

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
                wait_time = float(retry_after) if retry_after and retry_after.isdigit() else initial_backoff * (2**attempt)
                time.sleep(wait_time)
                continue
            if r.status_code == 200:
                return r
        except Exception:
            time.sleep(initial_backoff * (2**attempt))
    return None


def _parse_pages_to_candidates(pages: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract candidate image metadata from MediaWiki pages dict."""
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

    return candidates


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
            time.sleep(2.0 * (2**attempt))
            continue
        try:
            data = resp.json()
            break
        except Exception:
            time.sleep(2.0 * (2**attempt))

    if data is None:
        return [], None

    pages = data.get("query", {}).get("pages", {})
    candidates = _parse_pages_to_candidates(pages)
    next_continue = data.get("continue")
    return candidates, next_continue


def check_category_exists_and_has_files(session: requests.Session, category: str) -> bool:
    """Check that a Wikimedia Commons category exists and has files."""
    cat_title = category if category.startswith("Category:") else f"Category:{category}"
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": cat_title,
        "prop": "categoryinfo",
        "format": "json",
    }
    resp = request_with_backoff(session, api_url, params=params)
    if resp is None:
        return False
    try:
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for _, pinfo in pages.items():
            cinfo = pinfo.get("categoryinfo", {})
            if cinfo.get("files", 0) > 0:
                return True
    except Exception:
        return False
    return False


def fetch_category_batch(
    session: requests.Session,
    category_name: str,
    continue_params: dict[str, Any] | None = None,
    max_retries: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Fetch one batch of up to 100 files from a category in Wikimedia Commons."""
    cat_title = category_name if category_name.startswith("Category:") else f"Category:{category_name}"
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": cat_title,
        "gcmtype": "file",
        "gcmlimit": 100,
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
            time.sleep(2.0 * (2**attempt))
            continue
        try:
            data = resp.json()
            break
        except Exception:
            time.sleep(2.0 * (2**attempt))

    if data is None:
        return [], None

    pages = data.get("query", {}).get("pages", {})
    candidates = _parse_pages_to_candidates(pages)
    next_continue = data.get("continue")
    return candidates, next_continue


def to_thumb_url(url: str, width: int = 1280) -> str:
    """Convert Wikimedia Commons upload URL to 1280px thumbnail on thumb.wikimedia.org."""
    if not url:
        return ""
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
    parser = argparse.ArgumentParser(description="LoRA training dataset builder from Wikimedia Commons with CLIP filtering")
    parser.add_argument("--config", default="configs/lora_themes.yaml", help="Path to themes YAML config")
    parser.add_argument("--metadata-out", default="docs/lora_dataset_metadata.csv", help="Path to metadata CSV output")
    parser.add_argument("--overview-out", default="outputs/lora_dataset_overview.jpg", help="Path to overview image")
    parser.add_argument("--rejected-out", default="outputs/lora_rejected_overview.jpg", help="Path to rejected overview image")
    parser.add_argument("--data-dir", default="data/lora_train", help="Base dataset output directory")
    parser.add_argument("--from-metadata", help="Rebuild dataset from metadata CSV without filtering")
    parser.add_argument("--pool-size", type=int, default=250, help="Target pool size per theme before CLIP scoring")
    parser.add_argument("--target-per-theme", type=int, default=60, help="Target images per theme after CLIP ranking")
    args = parser.parse_args()

    if args.from_metadata:
        run_repro_mode(args.from_metadata, base_out_dir=args.data_dir)
        return

    with open(args.config, "r", encoding="utf-8") as f:
        themes_cfg = yaml.safe_load(f)

    session = requests.Session()
    session.headers.update({"User-Agent": "indic-product-photo-studio course project, student use"})

    base_data_path = Path(args.data_dir)
    # Wipe data/lora_train and rebuild
    if base_data_path.exists():
        shutil.rmtree(base_data_path, ignore_errors=True)
    base_data_path.mkdir(parents=True, exist_ok=True)

    all_accepted_hashes: list[imagehash.ImageHash] = []
    all_metadata_records: list[dict[str, Any]] = []

    stats: dict[str, dict[str, Any]] = {}
    rejected_by_theme: dict[str, list[np.ndarray]] = {}

    for theme, theme_info in themes_cfg.items():
        trigger = theme_info["trigger"]
        description = theme_info["description"]
        main_terms = theme_info.get("terms", [])
        fallback_terms = theme_info.get("fallback_terms", [])
        categories = theme_info.get("categories", [])
        pos_prompts = theme_info.get("clip_positive", [])
        neg_prompts = theme_info.get("clip_negative", [])

        theme_dir = base_data_path / theme
        theme_dir.mkdir(parents=True, exist_ok=True)

        theme_stats = {
            "candidates_found": 0,
            "dropped_person": 0,
            "dropped_dark_small": 0,
            "dropped_duplicate": 0,
            "dropped_author_cap": 0,
            "pool_size": 0,
            "passed_clip": 0,
            "final_count": 0,
            "final_threshold": 0.6,
            "mean_p_pos": 0.0,
        }
        theme_author_counts: dict[str, int] = defaultdict(int)
        theme_pool: list[dict[str, Any]] = []

        print(f"\n=======================================================")
        print(f"Processing theme: {theme} (trigger: '{trigger}')")
        print(f"Target pool size: {args.pool_size}")
        print(f"=======================================================")

        def process_candidate_item(cand: dict[str, Any], query_str: str) -> bool:
            if len(theme_pool) >= args.pool_size:
                return False

            theme_stats["candidates_found"] += 1

            img = download_candidate_image(session, cand["download_url"])
            if img is None:
                return False

            # FILTER 1: Person detection (conf=0.25, min_box_area=0.005)
            if contains_person(img, conf=0.25, min_box_area=0.005, ignore_mask=None):
                theme_stats["dropped_person"] += 1
                return False

            # FILTER 2: Dark or low-res (luminance < 40 or short side < 600)
            if check_dark_or_low_res(img, min_luminance=40.0, min_short_side=600):
                theme_stats["dropped_dark_small"] += 1
                return False

            # FILTER 3: Near-duplicates (phash dist <= 6 across all themes)
            phash = compute_phash(img)
            if is_duplicate(phash, all_accepted_hashes, max_distance=6):
                theme_stats["dropped_duplicate"] += 1
                return False

            # FILTER 4: Author diversity (author cap raised to 8 per theme)
            author = cand["author"]
            if theme_author_counts[author] >= 8:
                theme_stats["dropped_author_cap"] += 1
                return False

            # Passed cheap filters -> add to theme pool!
            theme_author_counts[author] += 1
            all_accepted_hashes.append(phash)
            theme_pool.append({
                "img": img,
                "cand": cand,
                "phash": phash,
                "author": author,
                "source_term": query_str,
            })
            theme_stats["pool_size"] = len(theme_pool)
            print(f"    [{theme}] Pooled {len(theme_pool)}/{args.pool_size}: {cand['title']}", flush=True)
            return True

        def evaluate_term(term_item: Any, is_fallback: bool = False):
            search_query = term_item["term"] if isinstance(term_item, dict) else str(term_item)
            tag = "[fallback]" if is_fallback else "[main]"
            print(f"  {tag} Query: '{search_query}'")

            continue_token = None
            pages_to_fetch = 5

            for _ in range(pages_to_fetch):
                if len(theme_pool) >= args.pool_size:
                    break
                candidates, continue_token = fetch_candidate_batch(session, search_query, continue_token)
                if not candidates:
                    break
                for cand in candidates:
                    process_candidate_item(cand, query_str=search_query)
                    if len(theme_pool) >= args.pool_size:
                        break
                if not continue_token:
                    break

        def evaluate_category(cat: str):
            print(f"  [category] Category: '{cat}'")
            if not check_category_exists_and_has_files(session, cat):
                print(f"WARNING: Category '{cat}' does not exist or has no files. Skipping.")
                return

            continue_token = None
            pages_to_fetch = 5

            for _ in range(pages_to_fetch):
                if len(theme_pool) >= args.pool_size:
                    break
                candidates, continue_token = fetch_category_batch(session, cat, continue_token)
                if not candidates:
                    break
                for cand in candidates:
                    process_candidate_item(cand, query_str=f"Category:{cat}")
                    if len(theme_pool) >= args.pool_size:
                        break
                if not continue_token:
                    break

        # 1. Main terms
        for term_item in main_terms:
            if len(theme_pool) >= args.pool_size:
                break
            evaluate_term(term_item, is_fallback=False)

        # 2. Categories
        for cat in categories:
            if len(theme_pool) >= args.pool_size:
                break
            evaluate_category(cat)

        # 3. Fallback terms (if pool < args.pool_size)
        if len(theme_pool) < args.pool_size:
            for fb_item in fallback_terms:
                if len(theme_pool) >= args.pool_size:
                    break
                evaluate_term(fb_item, is_fallback=True)

        print(f"\n  Theme '{theme}' collected pool of {len(theme_pool)} candidates.")

        # CLIP scoring
        if theme_pool and pos_prompts:
            print(f"  CLIP-scoring {len(theme_pool)} candidates on CPU in batches of 16...")
            clip_results = score_images_with_clip(
                images_bgr=[item["img"] for item in theme_pool],
                positive_prompts=pos_prompts,
                negative_prompts=neg_prompts,
                batch_size=16,
            )
            for item, (p_pos, best_prompt, best_caption) in zip(theme_pool, clip_results):
                item["clip_p_pos"] = p_pos
                item["clip_best_positive"] = best_prompt
                item["clip_best_caption"] = best_caption

            # Filter, relax threshold if needed, rank, and take top 60
            kept, final_thresh, passed_cnt = filter_and_rank_by_clip(
                theme_pool,
                min_count=40,
                top_k=args.target_per_theme,
                thresholds=(0.6, 0.5, 0.4),
            )

            theme_stats["passed_clip"] = passed_cnt
            theme_stats["final_count"] = len(kept)
            theme_stats["final_threshold"] = final_thresh
            theme_stats["mean_p_pos"] = float(np.mean([x["clip_p_pos"] for x in kept])) if kept else 0.0

            print(f"  Theme '{theme}' kept {len(kept)} images (threshold={final_thresh}, mean p_pos={theme_stats['mean_p_pos']:.3f})")

            # Collect 12 lowest p_pos for rejected contact sheet
            sorted_by_p_pos_asc = sorted(theme_pool, key=lambda x: x.get("clip_p_pos", 0.0))
            rejected_by_theme[theme] = [x["img"] for x in sorted_by_p_pos_asc[:12]]

            # Save kept images and captions
            for idx, item in enumerate(kept, start=1):
                crop = process_and_crop(item["img"], target_size=512)
                img_filename = f"{idx:03d}.jpg"
                txt_filename = f"{idx:03d}.txt"

                out_img_path = theme_dir / img_filename
                out_txt_path = theme_dir / txt_filename

                cv2.imwrite(str(out_img_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                caption_text = build_caption(trigger, item["clip_best_caption"], description)
                out_txt_path.write_text(caption_text, encoding="utf-8")

                all_metadata_records.append({
                    "filename": img_filename,
                    "theme": theme,
                    "term": item["source_term"],
                    "caption": caption_text,
                    "source_url": item["cand"]["source_url"],
                    "page_url": item["cand"]["page_url"],
                    "author": item["author"],
                    "license": item["cand"]["license"],
                    "phash": str(item["phash"]),
                    "clip_p_pos": round(item["clip_p_pos"], 4),
                    "clip_best_positive": item["clip_best_positive"],
                })
        else:
            theme_stats["passed_clip"] = 0
            theme_stats["final_count"] = 0
            theme_stats["final_threshold"] = 0.6
            theme_stats["mean_p_pos"] = 0.0

        stats[theme] = theme_stats

    # Save metadata CSV
    save_metadata_csv(all_metadata_records, args.metadata_out)
    print(f"\nMetadata saved to {args.metadata_out} ({len(all_metadata_records)} rows)")

    # Save overview sheets
    generate_overview_sheet(
        dataset_dir=args.data_dir,
        output_path=args.overview_out,
        themes=list(themes_cfg.keys()),
        seed=42,
    )
    print(f"Overview contact sheet saved to {args.overview_out}")

    generate_overview_sheet_from_images(
        theme_images=rejected_by_theme,
        output_path=args.rejected_out,
        themes=list(themes_cfg.keys()),
    )
    print(f"Rejected overview contact sheet saved to {args.rejected_out}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Theme':<16} {'Pool Size':<12} {'Passed CLIP':<14} {'Final Count':<12} {'Final Thresh':<14} {'Mean p_pos':<12}")
    print("-" * 90)
    for theme, s in stats.items():
        print(
            f"{theme:<16} {s['pool_size']:<12} {s['passed_clip']:<14} {s['final_count']:<12} "
            f"{s['final_threshold']:<14.1f} {s['mean_p_pos']:<12.3f}"
        )
    print("=" * 90)

    for theme, s in stats.items():
        if s["final_count"] < 40:
            print(f"WARNING: Theme '{theme}' has only {s['final_count']} images (< 40)!")


if __name__ == "__main__":
    main()
