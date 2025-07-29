#!/usr/bin/env python3
"""
video_captionator.py
Burns perfectly‐timed captions into a video (1280×720 example) and preserves audio.
All user‐editable settings are in the SETTINGS block below.
"""

import tempfile
from pathlib import Path

import numpy as np
import whisper
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------
# SETTINGS – change only these values
# ----------------------------------------------------------------------
INPUT_VIDEO_PATH   = Path("./myvideo.mp4")          # your source file
OUTPUT_VIDEO_PATH  = Path("./myvideo_captioned.mp4")# final result
WORD_LIMIT         = 2                              # words shown at once
TEXT_COLOR_RGB     = (255, 255, 0)                  # yellow
FONT_SIZE          = 64
FONT_FILE          = Path("./font.otf")             # .ttf / .otf
CAPTION_LOCATION_X = 640                            # pixels from LEFT edge
CAPTION_LOCATION_Y = 360                            # pixels from TOP  edge
WHISPER_MODEL      = "base"                         # whisper model choosing
# ----------------------------------------------------------------------

def split_segments_by_word_limit(segments, word_limit):
    """Yield {'start': float, 'end': float, 'text': str}."""
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

def text_to_array(text, font_obj, color_rgb, canvas_size):
    """Return H×W×4 uint8 NumPy array with text at exact CAPTION_LOCATION."""
    w, h = canvas_size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))  # fully transparent
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), text, font=font_obj)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # centre the text on the desired anchor point
    x = CAPTION_LOCATION_X - tw // 2
    y = CAPTION_LOCATION_Y - th // 2
    draw.text((x, y), text, font=font_obj, fill=(*color_rgb, 255))

    return np.array(img)

def main():
    # 1. Transcribe with Whisper
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(
        str(INPUT_VIDEO_PATH),
        word_timestamps=True,
        language=None
    )
    segments = list(split_segments_by_word_limit(result["segments"], WORD_LIMIT))

    # 2. Load video (with its audio)
    video = VideoFileClip(str(INPUT_VIDEO_PATH))

    # 3. Prepare the font
    font = ImageFont.truetype(str(FONT_FILE), FONT_SIZE)

    # 4. Build transparent ImageClip captions
    caption_clips = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        frame = text_to_array(text, font, TEXT_COLOR_RGB, (video.w, video.h))
        clip = (
            ImageClip(frame, duration=seg["end"] - seg["start"])
            .set_start(seg["start"])
            .set_position((0, 0))
        )
        caption_clips.append(clip)

    # 5. Composite captions onto the original video (it already has audio)
    composed = CompositeVideoClip([video, *caption_clips], size=(video.w, video.h))

    # 6. Write out with both video + audio intact
    composed.write_videofile(
        str(OUTPUT_VIDEO_PATH),
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(Path(tempfile.gettempdir()) / "temp-audio.m4a"),
        remove_temp=True,
        fps=video.fps
    )

    print("✅ Captioning complete →", OUTPUT_VIDEO_PATH)

if __name__ == "__main__":
    main()
