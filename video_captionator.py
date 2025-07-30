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

    # Apply delay before out to segments
    adjusted_segments = []
    for i, seg in enumerate(segments):
        if i < len(segments) - 1:
            next_start = segments[i + 1]['start']
            current_end = seg['end']
            # Extend the end time by the delay, but not beyond the next segment start
            extended_end = current_end + DELAY_BEFORE_OUT / 1000.0
            if extended_end < next_start:
                seg['end'] = extended_end
            else:
                seg['end'] = next_start
        adjusted_segments.append(seg)
    
    segments = adjusted_segments

    # Keep track of current segment and its display state
    current_segment_idx = 0
    current_text = ""
    display_until = -1  # Time until which current text should be displayed

    def overlay(get_frame, t):
        nonlocal current_segment_idx, current_text, display_until
        
        frame = get_frame(t)
        
        # Find the appropriate segment to display
        # First, move past segments that have definitely ended
        while current_segment_idx < len(segments) and t >= segments[current_segment_idx]['end']:
            current_segment_idx += 1
        
        # Check if we need to update the current text
        if current_segment_idx < len(segments):
            segment = segments[current_segment_idx]
            
            # If we're within the segment's original time, show it and update display_until
            if segment['start'] <= t < segment['end']:
                current_text = segment['text']
                display_until = segment['end']
            # If we're past the original end but within the extended time, keep showing the text
            elif t >= segment['end'] and t < display_until:
                # Keep showing the current text
                pass
            # If we're past the display_until time, clear the text
            elif t >= display_until:
                current_text = ""
        else:
            # No more segments, clear text
            current_text = ""
        
        # Render text if there's any to show
        if current_text:
            text_img = make_text_image(
                current_text, font, canvas_size, TEXT_COLOR_RGB,
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
TEXT_COLOR_RGB        = (255, 255, 255)  # white
FONT_SIZE             = 64
FONT_FILE             = Path("font.otf")
CAPTION_LOCATION_X    = 640
CAPTION_LOCATION_Y    = 360
WHISPER_MODEL         = "base"

# TEXT OUTLINE FEATURE
TEXT_OUTLINE_ENABLE   = True
TEXT_OUTLINE_COLOR    = (0, 0, 0)  # Black
TEXT_OUTLINE_SIZE     = 3  # pixels

# DELAY BEFORE OUT FEATURE
DELAY_BEFORE_OUT      = 1000  # milliseconds
# --------------------------------------

if __name__ == "__main__":
    main()