import openai
import os
import logging
logger = logging.getLogger(__name__)

def transcribe_audio(user, file_path):
    """
    Transcribe an audio file using OpenAI Whisper API.

    Parameters:
    - user: The user object, optionally used for custom OpenAI API keys.
    - file_path: Path to the audio file (.mp3, .wav, etc.)

    Returns:
    - The transcribed text.
    """
    # Optionally support per-user OpenAI key
    openai.api_key = user.openai_api_key_decrypted

    with open(file_path, "rb") as audio_file:
        logger.info(f"Transcribing {file_path} for user {user.id}")
        response = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
        transcript = response.text  # Access the text attribute
        logger.info("Transcription succeeded, %d characters", len(transcript))

    
    return transcript
