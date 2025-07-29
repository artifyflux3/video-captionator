import tempfile
from pathlib import Path
import numpy as np
import whisper
from moviepy.editor import VideoFileClip, CompositeVideoClip
from moviepy.video.VideoClip import ImageClip
from PIL import Image, ImageDraw, ImageFont

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
def text_to_array(text, font_obj, color_rgb, canvas_size):
    """Return H×W×4 uint8 NumPy array (RGBA) with text only."""
    w, h = canvas_size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font_obj)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = CAPTION_LOCATION_X - tw // 2
    y = CAPTION_LOCATION_Y - th // 2
    draw.text((x, y), text, font=font_obj, fill=color_rgb)
    return np.array(img)
def main():
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(str(INPUT_VIDEO_PATH), word_timestamps=True, language=None)
    segments = list(split_segments_by_word_limit(result["segments"], WORD_LIMIT))
    video = VideoFileClip(str(INPUT_VIDEO_PATH))
    font = ImageFont.truetype(str(FONT_FILE), FONT_SIZE)
    caption_clips = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        rgba_frame = text_to_array(text, font, TEXT_COLOR_RGB, (video.w, video.h))
        clip = (
            ImageClip(rgba_frame, ismask=False, duration=seg["end"] - seg["start"])
            .set_start(seg["start"])
            .set_position((0, 0))
        )
        caption_clips.append(clip)
    final = CompositeVideoClip([video] + caption_clips)
    final.write_videofile(
        str(OUTPUT_VIDEO_PATH),
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(OUTPUT_VIDEO_PATH.with_name("temp-audio.m4a")),
        remove_temp=True,
    )
    print("Captioning complete.")

# --------------------------------------
# SETTINGS
# --------------------------------------
INPUT_VIDEO_PATH   = Path("input.mp4")
OUTPUT_VIDEO_PATH  = Path("output.mp4")
WORD_LIMIT         = 2
TEXT_COLOR_RGB     = (255, 255, 0)
FONT_SIZE          = 64
FONT_FILE          = Path("font.otf")
CAPTION_LOCATION_X = 640
CAPTION_LOCATION_Y = 360
WHISPER_MODEL      = "base"
# --------------------------------------

if __name__ == "__main__":
    main()