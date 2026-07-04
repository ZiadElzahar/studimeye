import logging
from agent.schemas import Ticket, AgentResult
from agent.router import Router
from agent.services import ToolService
from agent.exceptions import AgentException, ToolExecutionError
from tools.text_tool import TextTool
from tools.image_tool import ImageTool
from tools.speech_tool import SpeechTool
from tools.recommendation_tool import RecommendationTool

logger = logging.getLogger(__name__)

class Orchestrator:
    """Main Agent entry point. Exposes only the process() method."""
    
    def __init__(self, timeout_seconds: int = 300):
        self.text_tool = TextTool()
        self.image_tool = ImageTool()
        self.speech_tool = SpeechTool()
        self.recommendation_tool = RecommendationTool()
        
        self.tool_service = ToolService(timeout_seconds=timeout_seconds)
        self.router = Router(
            self.tool_service, 
            self.text_tool, 
            self.image_tool,
            self.speech_tool
        )

    def process(self, ticket: Ticket) -> AgentResult:
        """Single public method to handle all routing and processing."""
        logger.info(f"Processing ticket {ticket.ticket_id}")
        
        result = AgentResult(ticket_id=ticket.ticket_id, status="processing")
        
        try:
            # 1. Route & Preprocess
            merged_text = self.router.route(ticket)
            result.extracted_text = merged_text
            
            # 2. Classification
            classification = self.tool_service.execute_tool(
                self.text_tool.process, merged_text
            )
            result.classification = classification
            
            # 3. Summarization/Recommendation
            summary = self.tool_service.execute_tool(
                self.recommendation_tool.process, merged_text
            )
            result.summary = summary
            
            result.status = "completed"
            
        except AgentException as e:
            logger.error(f"Agent failed to process ticket {ticket.ticket_id}: {e}")
            result.status = "failed"
            result.error = str(e)
        except Exception as e:
            logger.critical(f"Unexpected error for ticket {ticket.ticket_id}: {e}")
            result.status = "failed"
            result.error = f"Unexpected error: {str(e)}"
            
        return result