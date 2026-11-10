import csv
import html
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests

SEARCH_TERMS = [
    "brass diya",
    "oil lamp diya",
    "rakhi bracelet",
    "ceramic coffee mug",
    "leather handbag",
    "backpack",
    "wristwatch",
    "sneakers shoes",
    "stainless steel water bottle",
    "table lamp",
    "scented candle",
    "ceramic vase",
]

USER_AGENT = "indic-product-photo-studio/1.0 (https://github.com/Bipul20000/indic-product-photo-studio; indic-product-photo-studio course project, student use; student@example.org) requests/2.34.2"
API_ENDPOINT = "https://commons.wikimedia.org/w/api.php"


def is_valid_license(lic: str) -> bool:
    """Check if the license is CC0, Public domain, CC BY, or CC BY-SA."""
    if not lic:
        return False
    norm = lic.strip().lower().replace("-", " ")
    tokens = norm.split()
    # Reject non-commercial or no-derivatives
    if "nc" in tokens or "nd" in tokens or "noncommercial" in norm:
        return False
    # Check for allowed licenses
    if "cc0" in tokens or "cc0" in norm:
        return True
    if "public domain" in norm or norm == "pd" or norm.startswith("pd "):
        return True
    if "cc by" in norm:
        return True
    return False


def clean_artist(artist_raw: str) -> str:
    """Strip HTML tags and unescape HTML entities from the artist field."""
    if not artist_raw:
        return "Unknown"
    text = re.sub(r"<[^>]+>", "", artist_raw)
    cleaned = html.unescape(text).strip()
    return cleaned if cleaned else "Unknown"


def sanitize_filename(title: str, max_length: int = 80) -> str:
    """Produce a safe, clean filename from a Commons title."""
    clean = title
    if clean.startswith("File:"):
        clean = clean[5:]
    stem = Path(clean).stem
    stem = re.sub(r"[^\w\-.]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._")
    if len(stem) > max_length:
        stem = stem[:max_length]
    return f"{stem}.jpg"


def request_with_retry(
    session: requests.Session,
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    stream: bool = False,
    max_retries: int = 4,
) -> Optional[requests.Response]:
    """Perform an HTTP request with exponential backoff on 429/5xx and rate limiting."""
    delay = 4.0
    for attempt in range(max_retries):
        try:
            time.sleep(1.2)
            resp = session.get(url, params=params, headers=headers, stream=stream, timeout=40)
            if resp.status_code == 429 or resp.status_code >= 500:
                print(f"HTTP {resp.status_code} received. Backing off for {delay:.1f}s (attempt {attempt + 1}/{max_retries})...", flush=True)
                time.sleep(delay)
                delay *= 2.0
                continue
            if resp.status_code == 403:
                # 403 is access forbidden by host policy, do not waste time retrying
                return None
            resp.raise_for_status()
            return resp
        except (requests.RequestException, Exception) as exc:
            if attempt == max_retries - 1:
                print(f"Request failed for {url}: {exc}", flush=True)
                return None
            time.sleep(delay)
            delay *= 2.0
    return None


def fetch_photos():
    output_dir = Path("data/coach_eval/originals/good")
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    attribution_csv = docs_dir / "coach_eval_attribution.csv"

    # Track already existing filenames on disk
    existing_files = {p.name: p for p in output_dir.iterdir() if p.is_file()}

    seen_titles = set()
    attribution_records: List[Dict[str, str]] = []
    if attribution_csv.exists():
        with open(attribution_csv, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                attribution_records.append(row)
                seen_titles.add(row.get("title", ""))

    session = requests.Session()
    headers = {"User-Agent": USER_AGENT}

    print(f"Starting photo fetch from Wikimedia Commons across {len(SEARCH_TERMS)} search terms...", flush=True)
    print(f"Already have {len(existing_files)} photos on disk and {len(attribution_records)} attribution records.", flush=True)

    fieldnames = ["filename", "title", "source_url", "author", "license"]

    def save_attribution():
        with open(attribution_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(attribution_records)

    for term_idx, term in enumerate(SEARCH_TERMS, start=1):
        print(f"\n--- [{term_idx}/{len(SEARCH_TERMS)}] Searching term: '{term}' ---", flush=True)
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": term,
            "gsrnamespace": 6,
            "gsrlimit": 20,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 1600,
            "format": "json",
        }

        resp = request_with_retry(session, API_ENDPOINT, params=params, headers=headers)
        if resp is None:
            print(f"Could not retrieve search results for '{term}'. Skipping.", flush=True)
            continue

        try:
            data = resp.json()
        except Exception as e:
            print(f"Error parsing JSON for '{term}': {e}", flush=True)
            continue

        pages = data.get("query", {}).get("pages", {})
        term_kept = 0

        for page_id, page in pages.items():
            if term_kept >= 6:
                break

            title = page.get("title", "")
            if title in seen_titles:
                continue

            imageinfo_list = page.get("imageinfo", [])
            if not imageinfo_list:
                continue
            info = imageinfo_list[0]

            mime = info.get("mime", "")
            if mime != "image/jpeg":
                continue

            orig_width = info.get("width", 0)
            if orig_width < 1000:
                continue

            extmetadata = info.get("extmetadata", {})
            license_val = extmetadata.get("LicenseShortName", {}).get("value", "")
            if not is_valid_license(license_val):
                continue

            download_url = info.get("thumburl") or info.get("url")
            if not download_url:
                continue

            # Format destination filename
            filename = sanitize_filename(title)
            dest_path = output_dir / filename

            source_url = info.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}"
            artist_val = extmetadata.get("Artist", {}).get("value", "Unknown")
            author = clean_artist(artist_val)

            if dest_path.exists():
                print(f"Existing on disk: {filename}", flush=True)
            else:
                img_resp = request_with_retry(session, download_url, headers=headers)
                # If downloading thumburl returned None or 403, fallback to 800px thumb
                if img_resp is None:
                    fallback_params = {
                        "action": "query",
                        "titles": title,
                        "prop": "imageinfo",
                        "iiprop": "url",
                        "iiurlwidth": 800,
                        "format": "json",
                    }
                    fb_resp = request_with_retry(session, API_ENDPOINT, params=fallback_params, headers=headers)
                    if fb_resp:
                        try:
                            fb_info = list(fb_resp.json().get("query", {}).get("pages", {}).values())[0]["imageinfo"][0]
                            fb_url = fb_info.get("thumburl") or fb_info.get("url")
                            if fb_url:
                                img_resp = request_with_retry(session, fb_url, headers=headers)
                        except Exception:
                            pass

                if img_resp is None:
                    print(f"Skipping (could not download image): {title}", flush=True)
                    continue

                with open(dest_path, "wb") as img_file:
                    img_file.write(img_resp.content)
                print(f"Downloaded [{term}] ({term_kept + 1}/6): {filename} (width={orig_width}, lic={license_val})", flush=True)

            seen_titles.add(title)
            attribution_records.append({
                "filename": filename,
                "title": title,
                "source_url": source_url,
                "author": author,
                "license": license_val,
            })
            save_attribution()
            term_kept += 1

    save_attribution()
    print(f"\nAttribution saved to: {attribution_csv}", flush=True)
    print(f"Total photos collected: {len(attribution_records)}", flush=True)

    if len(attribution_records) < 40:
        print(f"\nWARNING: Fewer than 40 images were downloaded (only {len(attribution_records)}).", flush=True)


if __name__ == "__main__":
    fetch_photos()
