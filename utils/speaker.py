from gtts import gTTS
import io

def speak(text, rate=150):
    """Generate speech audio bytes from text using gTTS. Returns audio bytes or None on failure."""
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.read()
    except Exception as e:
        print(f"⚠️ Voice error: {e}")
        return None

def describe_scene(detections):
    if not detections:
        return "I don't see anything ahead."

    objects = [d['label'] for d in detections if d['confidence'] > 0.5]
    unique_objects = list(set(objects))

    if not unique_objects:
        return "Nothing detected clearly."
    elif len(unique_objects) == 1:
        return f"There is a {unique_objects[0]} ahead."
    elif len(unique_objects) <= 3:
        return f"There are {', '.join(unique_objects)} ahead."
    else:
        return f"There are multiple objects: {', '.join(unique_objects[:3])} and more."