import re
import os
import tempfile
from moviepy.editor import (
    VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip,
    concatenate_videoclips, CompositeAudioClip, ImageClip,
    concatenate_audioclips, ColorClip,
)
import numpy as np
import PIL.Image
from PIL import Image, ImageDraw, ImageFont
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

TARGET_W, TARGET_H = 1080, 1920
FONT_FILES = [
    os.environ.get("CAPTION_FONT_FILE", ""),
    "/usr/share/fonts/truetype/noto/NotoSansTelugu-Bold.ttf",
    os.path.expanduser("~/.fonts/NotoSansTelugu-Bold.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _load_font(size: int):
    for path in FONT_FILES:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_text(draw, text: str, font, max_width: int) -> str:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if not cur or draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _render_caption_image(text: str, max_width: int, fontsize: int):
    font = _load_font(fontsize)
    probe = Image.new("RGBA", (10, 10))
    d = ImageDraw.Draw(probe)
    wrapped = _wrap_text(d, text, font, max_width)
    bbox = d.multiline_textbbox((0, 0), wrapped, font=font, spacing=12, align="center", stroke_width=2)
    w, h = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    pad = 14
    img = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(img)
    d2.multiline_text(
        (int(pad - bbox[0]), int(pad - bbox[1])), wrapped, font=font, fill="white",
        spacing=12, align="center", stroke_width=2, stroke_fill="black",
    )
    return img
IMAGE_DURATION = 3.0
CAPTION_Y = TARGET_H - 600


def _split_phrases(text: str) -> list[str]:
    text = text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    phrases = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s.split()) > 6:
            parts = re.split(r"(?<=[,;:])\s+", s)
            phrases.extend(p.strip() for p in parts if p.strip())
        else:
            phrases.append(s)
    return phrases


def _estimate_timings(phrases: list[str], total_duration: float) -> list[tuple[str, float]]:
    if not phrases:
        return []
    word_counts = [len(p.split()) for p in phrases]
    total_words = sum(word_counts)
    if total_words == 0:
        return [(phrases[0], total_duration)]
    return [(p, (wc / total_words) * total_duration) for p, wc in zip(phrases, word_counts)]


def _crop_to_portrait(clip):
    w, h = clip.size
    target_ratio = TARGET_W / TARGET_H
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x_center = w // 2
        clip = clip.crop(x1=x_center - new_w // 2, x2=x_center + new_w // 2)
    elif current_ratio < target_ratio:
        new_h = int(w / target_ratio)
        y_center = h // 2
        clip = clip.crop(y1=y_center - new_h // 2, y2=y_center + new_h // 2)
    return clip.resize((TARGET_W, TARGET_H))





def _interleave(video_clips: list, image_clips: list) -> list:
    if not image_clips:
        return list(video_clips)
    result = []
    gap = max(1, len(video_clips) // len(image_clips))
    img_idx = 0
    for i, v in enumerate(video_clips):
        if img_idx < len(image_clips) and i > 0 and i % gap == 0:
            result.append(image_clips[img_idx])
            img_idx += 1
        result.append(v)
    while img_idx < len(image_clips):
        result.append(image_clips[img_idx])
        img_idx += 1
    return result


def _make_caption(text: str, start: float, dur: float):
    # PIL + libraqm renders complex scripts (e.g. Telugu) with correct shaping,
    # unlike ImageMagick TextClip which breaks conjunct glyphs.
    img = _render_caption_image(text, TARGET_W - 200, 60)
    txt = ImageClip(np.array(img), transparent=True)
    txt_w, txt_h = txt.size
    bg_box = ColorClip(size=(txt_w + 40, txt_h + 20), color=(0, 0, 0))
    bg_box = bg_box.set_opacity(0.7)
    bg_box = bg_box.set_position(("center", CAPTION_Y - 10))
    txt = txt.set_position(("center", CAPTION_Y))
    caption = CompositeVideoClip([bg_box, txt], size=(TARGET_W, TARGET_H))
    caption = caption.set_start(start).set_duration(dur)
    return caption


class VideoEditor:
    def __init__(self, workdir: str | None = None):
        self.workdir = workdir or tempfile.mkdtemp(prefix="shorts_edit_")

    def compose(
        self,
        video_paths: list[str],
        audio_path: str,
        script: str,
        output_path: str,
        audio_duration: float,
        image_paths: list[str] | None = None,
        music_path: str | None = None,
        sfx_tracks: list[tuple[str, float]] | None = None,
    ) -> str:
        image_paths = image_paths or []

        print("Loading video clips...")
        video_clips = []
        for path in video_paths:
            try:
                clip = VideoFileClip(path)
                clip = _crop_to_portrait(clip)
                clip = clip.set_duration(min(clip.duration, 10))
                video_clips.append(clip)
            except Exception as e:
                print(f"  Failed to load {path}: {e}")
        if not video_clips:
            raise RuntimeError("No usable video clips to compose")

        print("Loading images...")
        image_clips = []
        for path in image_paths:
            try:
                clip = ImageClip(path)
                clip = _crop_to_portrait(clip)
                clip = clip.set_duration(IMAGE_DURATION)
                image_clips.append(clip)
            except Exception as e:
                print(f"  Failed to load image {path}: {e}")

        print("Interleaving clips...")
        segments = _interleave(video_clips, image_clips)

        total_raw = sum(c.duration for c in segments)
        if total_raw < audio_duration:
            factor = int(audio_duration / total_raw) + 1
            segments = segments * factor

        trimmed = []
        accumulated = 0.0
        for c in segments:
            needed = audio_duration - accumulated
            if needed <= 0:
                c.close()
                break
            if c.duration > needed:
                trimmed.append(c.subclip(0, needed))
                accumulated += needed
            else:
                trimmed.append(c)
                accumulated += c.duration

        print(f"  {len(trimmed)} segments ({accumulated:.1f}s total)")

        print("Concatenating video...")
        final_video = concatenate_videoclips(trimmed, method="chain")

        print("Mixing audio...")
        voiceover = AudioFileClip(audio_path)
        audio_tracks = [voiceover]

        if music_path and os.path.isfile(music_path):
            try:
                bg_music = AudioFileClip(music_path)
                if bg_music.duration < audio_duration:
                    n = int(audio_duration / bg_music.duration) + 1
                    bg_music = concatenate_audioclips([bg_music] * n)
                    bg_music = bg_music.subclip(0, audio_duration)
                bg_music = bg_music.set_duration(audio_duration)
                bg_music = bg_music.volumex(0.08)  # Duck BGM slightly lower for clear narrator voiceover
                audio_tracks.append(bg_music)
            except Exception as e:
                print(f"  Failed to load background music: {e}")

        # Mix sound effects at specified timestamps
        sfx_tracks = sfx_tracks or []
        for sfx_path, timestamp in sfx_tracks:
            try:
                if os.path.isfile(sfx_path):
                    sfx_clip = AudioFileClip(sfx_path)
                    sfx_clip = sfx_clip.set_duration(min(sfx_clip.duration, 5.0))
                    sfx_clip = sfx_clip.volumex(0.8).set_start(timestamp)
                    audio_tracks.append(sfx_clip)
                    print(f"  Mixed sound effect '{os.path.basename(sfx_path)}' at {timestamp:.2f}s")
            except Exception as e:
                print(f"  Failed to mix sound effect {sfx_path}: {e}")

        final_audio = CompositeAudioClip(audio_tracks)
        final_video = final_video.set_audio(final_audio)

        print("Generating captions...")
        phrases = _split_phrases(script)
        timings = _estimate_timings(phrases, audio_duration)

        subtitle_clips = []
        current_time = 0.0
        for text, dur in timings:
            sub = _make_caption(text, current_time, dur)
            subtitle_clips.append(sub)
            current_time += dur

        print("Compositing final video...")
        final = CompositeVideoClip([final_video, *subtitle_clips], size=(TARGET_W, TARGET_H))

        print("Rendering...")
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            preset="ultrafast",
            threads=2,
            bitrate="3000k",
            logger=None,
        )

        for c in trimmed:
            c.close()
        for t in subtitle_clips:
            t.close()
        for track in audio_tracks:
            try:
                track.close()
            except Exception:
                pass
        final_video.close()
        final.close()

        return output_path
