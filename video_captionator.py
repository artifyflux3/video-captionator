import json
from pathlib import Path
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
import whisper

def split_segments_by_word_limit(segments, word_limit):
    for seg in segments:
        words = seg.get("words", [])
        if not words:
            continue
        for i in range(0, len(words), word_limit):
            chunk = words[i : i + word_limit]
            start = chunk[0]["start"]
            end = chunk[-1]["end"]
            text = " ".join(w["word"].strip() for w in chunk)
            yield {"start": start, "end": end, "text": text}

def write_jsonl(segments, path):
    with path.open("w", encoding="utf-8") as f:
        for seg in segments:
            json.dump(seg, f, ensure_ascii=False)
            f.write("\n")

def load_segments(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def make_text_image(text, font, canvas_size, color, outline_enable, outline_color, outline_size):
    w, h = canvas_size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = CAPTION_LOCATION_X - tw // 2
    y = CAPTION_LOCATION_Y - th // 2

    if outline_enable:
        # Draw outline by drawing text multiple times with offset
        for dx in range(-outline_size, outline_size + 1):
            for dy in range(-outline_size, outline_size + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

    # Draw main text on top
    draw.text((x, y), text, font=font, fill=color)
    return img

def main():
    # 1. Transcribe and write captions
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(str(INPUT_VIDEO_PATH), word_timestamps=True)
    segments_gen = split_segments_by_word_limit(result["segments"], WORD_LIMIT)
    write_jsonl(segments_gen, CAPTIONS_JSONL)

    # 2. Load captions for overlay
    segments = load_segments(CAPTIONS_JSONL)
    video = VideoFileClip(str(INPUT_VIDEO_PATH))
    font = ImageFont.truetype(str(FONT_FILE), FONT_SIZE)
    canvas_size = (video.w, video.h)

    # prepare segment iterator
    segments = sorted(segments, key=lambda s: s['start'])
    idx = 0

    def overlay(get_frame, t):
        nonlocal idx
        frame = get_frame(t)
        # advance idx if past current segment
        while idx < len(segments) and t >= segments[idx]['end']:
            idx += 1
        # draw if within current segment
        if idx < len(segments) and segments[idx]['start'] <= t < segments[idx]['end']:
            text = segments[idx]['text']
            text_img = make_text_image(
                text, font, canvas_size, TEXT_COLOR_RGB,
                TEXT_OUTLINE_ENABLE, TEXT_OUTLINE_COLOR, TEXT_OUTLINE_SIZE
            )
            # composite PIL over frame
            pil_frame = Image.fromarray(frame)
            pil_frame.paste(text_img, (0, 0), text_img)
            return np.array(pil_frame)
        return frame

    # apply overlay without storing all clips
    processed = video.fl(overlay, apply_to=['video'])
    processed.write_videofile(
        str(OUTPUT_VIDEO_PATH),
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(OUTPUT_VIDEO_PATH.with_name("temp-audio.m4a")),
        remove_temp=True,
    )
    print("Captioning complete. Captions written to", CAPTIONS_JSONL)

# --------------------------------------
# SETTINGS
# --------------------------------------
INPUT_VIDEO_PATH      = Path("input.mp4")
OUTPUT_VIDEO_PATH     = Path("output.mp4")
CAPTIONS_JSONL        = Path("captions.jsonl")
WORD_LIMIT            = 2
TEXT_COLOR_RGB        = (255, 255, 0)  # Yellow
FONT_SIZE             = 64
FONT_FILE             = Path("font.otf")
CAPTION_LOCATION_X    = 640
CAPTION_LOCATION_Y    = 360
WHISPER_MODEL         = "base"

# TEXT OUTLINE FEATURE
TEXT_OUTLINE_ENABLE   = True
TEXT_OUTLINE_COLOR    = (0, 0, 0)  # Black
TEXT_OUTLINE_SIZE     = 3  # pixels
# --------------------------------------

if __name__ == "__main__":
    main()