import logging
from agent.schemas import Ticket
from agent.services import ToolService
from agent.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)

class Router:
    """Routes traffic based on modality presence."""
    
    def __init__(self, tool_service: ToolService, text_tool, image_tool, speech_tool):
        self.tool_service = tool_service
        self.text_tool = text_tool
        self.image_tool = image_tool
        self.speech_tool = speech_tool

    def route(self, ticket: Ticket) -> str:
        """Determines and executes the preprocessing pipeline, returning merged text."""
        extracted_texts = []

        # Process Audio
        if ticket.audio_path:
            if self.speech_tool is not None:
                try:
                    transcript = self.tool_service.execute_tool(
                        self.speech_tool.process, ticket.audio_path
                    )
                    extracted_texts.append(f"[Audio Transcript]: {transcript}")
                except ToolExecutionError as e:
                    logger.warning(f"Skipping audio modality due to error: {e}")
            else:
                logger.warning("Skipping audio modality: SpeechTool is not initialized.")

        # Process Image
        if ticket.image_path:
            if self.image_tool is not None:
                try:
                    ocr_text = self.tool_service.execute_tool(
                        self.image_tool.process, ticket.image_path
                    )
                    extracted_texts.append(f"[Image Extracted Text]: {ocr_text}")
                except ToolExecutionError as e:
                    logger.warning(f"Skipping image modality due to error: {e}")
            else:
                logger.warning("Skipping image modality: ImageTool is not initialized.")

        # Process Text
        if ticket.text:
            extracted_texts.append(f"[Original Text]: {ticket.text}")

        if not extracted_texts:
            raise ToolExecutionError("All modalities failed to process or were skipped. No text to classify.")

        return "\n".join(extracted_texts)