import logging
import concurrent.futures
from typing import Callable, Any
from agent.exceptions import ToolTimeoutError, ToolExecutionError

logger = logging.getLogger(__name__)

class ToolService:
    """Service layer to execute tools with timeouts and error handling."""
    
    def __init__(self, timeout_seconds: int = 120):
        self.timeout = timeout_seconds

    def execute_tool(self, func: Callable, *args, **kwargs) -> Any:
        """Executes a tool function in a separate thread with a timeout."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=self.timeout)
                return result
            except concurrent.futures.TimeoutError:
                logger.error(f"Tool {func.__name__} timed out after {self.timeout}s.")
                raise ToolTimeoutError(f"Execution timed out for {func.__name__}")
            except Exception as e:
                logger.error(f"Tool {func.__name__} failed: {str(e)}")
                raise ToolExecutionError(f"Tool execution failed: {str(e)}")