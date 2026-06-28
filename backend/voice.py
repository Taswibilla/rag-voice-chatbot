import io
import base64
import speech_recognition as sr
from gtts import gTTS


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Convert recorded audio bytes to text.
    Uses Google Speech Recognition (free, no API key needed).
    """
    recognizer = sr.Recognizer()
    
    audio_io = io.BytesIO(audio_bytes)
    
    try:
        with sr.AudioFile(audio_io) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
        
        text = recognizer.recognize_google(audio_data)
        return text
    
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio. Please try again."
    except sr.RequestError as e:
        return f"Speech recognition error: {e}"


def text_to_speech(text: str, lang: str = "en") -> str:
    """
    Convert text to speech audio.
    Returns base64-encoded MP3 string for easy transport.
    """
    if len(text) > 3000:
        text = text[:3000] + "... (response truncated)"
    
    tts = gTTS(text=text, lang=lang, slow=False)
    
    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)
    
    audio_b64 = base64.b64encode(mp3_buffer.read()).decode("utf-8")
    return audio_b64