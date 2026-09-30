import os
import requests
import tempfile
from pathlib import Path
import hashlib

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY")
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY")

PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"
PEXELS_HEADERS = {"Authorization": PEXELS_API_KEY}

PIXABAY_VIDEO_URL = "https://pixabay.com/api/videos/"
PIXABAY_PHOTO_URL = "https://pixabay.com/api/"

MAX_CLIPS = 10
MAX_IMAGES = 8
MAX_PER_KEYWORD = 5


class VideoDownloader:
    def __init__(self):
        if PIXABAY_API_KEY:
            self.provider = "pixabay"
        elif PEXELS_API_KEY:
            self.provider = "pexels"
        else:
            raise ValueError("Set PIXABAY_API_KEY or PEXELS_API_KEY environment variable")
        print(f"  Media provider: {self.provider}")

    # ---------- Pixabay ----------

    def _pixabay_clips(self, keywords: list[str]) -> list[dict]:
        seen = set()
        clips = []
        for kw in keywords:
            params = {"key": PIXABAY_API_KEY, "q": kw, "per_page": MAX_PER_KEYWORD}
            try:
                resp = requests.get(PIXABAY_VIDEO_URL, params=params, timeout=15)
                resp.raise_for_status()
            except Exception as e:
                print(f"  Pixabay video search failed for '{kw}': {e}")
                continue
            for hit in resp.json().get("hits", []):
                vid_id = hit["id"]
                if vid_id in seen:
                    continue
                seen.add(vid_id)
                variants = hit.get("videos", {})
                chosen = None
                for size in ("large", "medium", "small"):
                    v = variants.get(size)
                    if v and v.get("url"):
                        chosen = v
                        break
                if not chosen:
                    continue
                clips.append({
                    "id": vid_id,
                    "url": chosen["url"],
                    "width": chosen.get("width", 0),
                    "height": chosen.get("height", 0),
                    "duration": hit.get("duration", 10),
                })
                if len(clips) >= MAX_CLIPS:
                    break
            if len(clips) >= MAX_CLIPS:
                break
        return clips[:MAX_CLIPS]

    def _pixabay_images(self, keywords: list[str]) -> list[dict]:
        seen = set()
        images = []
        for kw in keywords:
            params = {
                "key": PIXABAY_API_KEY, "q": kw, "per_page": MAX_PER_KEYWORD,
                "image_type": "photo", "orientation": "vertical",
            }
            try:
                resp = requests.get(PIXABAY_PHOTO_URL, params=params, timeout=15)
                resp.raise_for_status()
            except Exception as e:
                print(f"  Pixabay photo search failed for '{kw}': {e}")
                continue
            for hit in resp.json().get("hits", []):
                pid = hit["id"]
                if pid in seen:
                    continue
                seen.add(pid)
                images.append({"id": pid, "url": hit.get("largeImageURL") or hit.get("webformatURL")})
                if len(images) >= MAX_IMAGES:
                    break
            if len(images) >= MAX_IMAGES:
                break
        return images[:MAX_IMAGES]

    # ---------- Pexels ----------

    def _pexels_clips(self, keywords: list[str]) -> list[dict]:
        seen = set()
        clips = []
        for kw in keywords:
            params = {"query": kw, "per_page": MAX_PER_KEYWORD, "orientation": "portrait"}
            resp = requests.get(PEXELS_VIDEO_URL, headers=PEXELS_HEADERS, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            for video in data.get("videos", []):
                vid_id = video["id"]
                if vid_id in seen:
                    continue
                seen.add(vid_id)
                for file in video.get("video_files", []):
                    if file["quality"] in ("sd", "hd") and file.get("link"):
                        clips.append({
                            "id": vid_id,
                            "url": file["link"],
                            "width": file.get("width", 0),
                            "height": file.get("height", 0),
                            "duration": video.get("duration", 10),
                        })
                        break
                if len(clips) >= MAX_CLIPS:
                    break
            if len(clips) >= MAX_CLIPS:
                break
        return clips[:MAX_CLIPS]

    def _pexels_images(self, keywords: list[str]) -> list[dict]:
        seen = set()
        images = []
        for kw in keywords:
            params = {"query": kw, "per_page": MAX_PER_KEYWORD, "orientation": "portrait"}
            resp = requests.get(PEXELS_PHOTO_URL, headers=PEXELS_HEADERS, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            for photo in data.get("photos", []):
                pid = photo["id"]
                if pid in seen:
                    continue
                seen.add(pid)
                images.append({
                    "id": pid,
                    "url": photo["src"]["large"],
                })
                if len(images) >= MAX_IMAGES:
                    break
            if len(images) >= MAX_IMAGES:
                break
        return images[:MAX_IMAGES]

    # ---------- Shared ----------

    def search_clips(self, keywords: list[str]) -> list[dict]:
        if self.provider == "pixabay":
            return self._pixabay_clips(keywords)
        return self._pexels_clips(keywords)

    def search_images(self, keywords: list[str]) -> list[dict]:
        if self.provider == "pixabay":
            return self._pixabay_images(keywords)
        return self._pexels_images(keywords)

    def download_file(self, url: str, output_dir: str, prefix: str) -> str | None:
        ext = Path(url.split("?")[0]).suffix or ".mp4"
        hash_str = hashlib.md5(url.encode()).hexdigest()[:12]
        out_path = os.path.join(output_dir, f"{prefix}_{hash_str}{ext}")
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return out_path
        except Exception as e:
            print(f"  Failed to download {prefix}_{hash_str}: {e}")
            return None

    def download_clip(self, clip: dict, output_dir: str) -> str | None:
        return self.download_file(clip["url"], output_dir, f"clip_{clip['id']}")

    def download_image(self, image: dict, output_dir: str) -> str | None:
        return self.download_file(image["url"], output_dir, f"img_{image['id']}")

    def download_all(self, keywords: list[str], output_dir: str | None = None):
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="shorts_")
        else:
            os.makedirs(output_dir, exist_ok=True)

        clips = self.search_clips(keywords)
        print(f"  Found {len(clips)} clips, downloading...")
        video_paths = []
        for clip in clips:
            path = self.download_clip(clip, output_dir)
            if path:
                video_paths.append(path)

        images = self.search_images(keywords)
        print(f"  Found {len(images)} images, downloading...")
        image_paths = []
        for img in images:
            path = self.download_image(img, output_dir)
            if path:
                image_paths.append(path)

        return video_paths, image_paths
