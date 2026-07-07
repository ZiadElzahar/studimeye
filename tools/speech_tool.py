import os
import logging
from transformers import pipeline
import dotenv

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Gemini import — safe fallback if not installed
try:
    import google.generativeai as genai
except ImportError:
    genai = None


class SpeechTool:
    def __init__(self):
        # --- Whisper (local ASR) ---
        model_name = "openai/whisper-base"
        print(f"Loading SpeechTool ({model_name}) locally on CPU...")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=-1
        )
        print("SpeechTool (Whisper) is ready!")

        # --- Gemini (transcript correction) ---
        self.gemini_model = None
        if genai is not None:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("Gemini 3.5 Flash")
                print("SpeechTool (Gemini Correction) is ready!")
            else:
                log.warning("No GEMINI_API_KEY found. Transcript correction is disabled.")
        else:
            log.warning("google-generativeai not installed. Transcript correction is disabled.")

    def _correct_transcript(self, raw_transcript: str) -> str:
        """Sends raw ASR text to Gemini to fix language mistakes."""
        if not self.gemini_model:
            return raw_transcript

        prompt = f"""You are a text correction assistant for automatic speech recognition (ASR) output.

Your ONLY job is to fix the following types of mistakes in the transcribed text:
- Spelling errors caused by misheard words
- Missing or wrong punctuation
- Grammatical mistakes
- Run-on sentences that need splitting
- Obvious wrong word choices (e.g. "their" vs "there")

STRICT RULES:
- Do NOT add any information that was not spoken.
- Do NOT change the meaning or tone.
- Do NOT summarize or shorten the text.
- Keep the original language. If the transcript is in Arabic, fix it in Arabic. If English, fix in English.
- Return ONLY the corrected text. No explanations, no prefixes, no quotes.

Raw Transcript:
\"\"\"{raw_transcript}\"\"\"

Corrected Text:"""

        try:
            response = self.gemini_model.generate_content(prompt)
            corrected = response.text.strip()
            log.info(f"Transcript corrected successfully.")
            return corrected
        except Exception as e:
            log.error(f"Gemini correction failed: {e}. Returning raw transcript.")
            return raw_transcript

    def process(self, audio_path: str) -> str:
        """Public method called by orchestrator. Returns corrected transcript string."""
        print(f"Processing audio file: {audio_path}... (This might take a few moments on CPU)")
        try:
            # Step 1: Transcribe
            output = self.pipe(audio_path)
            raw_transcript = output["text"].strip()
            print(f"Raw Transcript: {raw_transcript}")

            # Step 2: Correct with Gemini
            corrected_transcript = self._correct_transcript(raw_transcript)
            print(f"Corrected Transcript: {corrected_transcript}")

            return corrected_transcript
        except Exception as exc:
            log.error("Whisper local execution failed: %s", exc)
            return f"Error processing audio: {str(exc)}"


if __name__ == "__main__":
    tool = SpeechTool()
    # test_audio_path = "test_audio.wav"
    # print("Extracted Text:", tool.process(test_audio_path))
    pass