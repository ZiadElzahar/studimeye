class AgentException(Exception):
    """Base exception for Agent."""
    pass

class ToolTimeoutError(AgentException):
    """Raised when a tool execution exceeds the time limit."""
    pass

class ToolExecutionError(AgentException):
    """Raised when a tool fails internally."""
    pass

class TicketValidationError(AgentException):
    """Raised when input ticket validation fails."""
    pass