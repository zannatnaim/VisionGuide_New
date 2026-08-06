import pyttsx3

def speak(text, rate=150):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.setProperty('volume', 0.9)
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"⚠️ Voice error: {e}")
        return False

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