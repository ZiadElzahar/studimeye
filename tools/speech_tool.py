import logging
from typing import Any
# Wrapped safely to avoid import crashes in non-Kaggle environments
try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import InferenceClient
except ImportError:
    UserSecretsClient = None
    InferenceClient = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class SpeechTool:
    def __init__(self):
        if UserSecretsClient and InferenceClient:
            user_secrets = UserSecretsClient()
            hf_token = user_secrets.get_secret("hftoken") 
            self.client = InferenceClient(provider="auto", api_key=hf_token)
        else:
            self.client = None
            log.warning("Kaggle Secrets or HuggingFace Hub not available. SpeechTool is disabled.")

    def process(self, audio_path: str) -> str:
        """Public method called by orchestrator. Returns transcript string."""
        if not self.client:
            raise RuntimeError("SpeechTool client is not initialized.")
        
        try:
            output: Any = self.client.automatic_speech_recognition(
                audio_path, model="openai/whisper-large-v3"
            )
            return str(output)
        except Exception as exc:
            log.error("Whisper ASR call failed: %s", exc)
            raise exc

if __name__ == "__main__":
    tool = SpeechTool()
    # output = tool.process("/kaggle/input/datasets/ziadelzahar/testvc/WhatsApp Ptt 2026-06-16 at 8.29.11 PM.ogg")
    # print(output)
    pass