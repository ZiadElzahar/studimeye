import logging
from agent.orchestrator import Orchestrator
from agent.schemas import Ticket

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Initialize Agent
    agent = Orchestrator(timeout_seconds=180)
    
    # Example 1: Only Text
    ticket_1 = Ticket(
        ticket_id="T-001",
        text="Someone just stole my backpack while I was waiting near Gate D. My passport and wallet are inside and my flight leaves in 2 hours!"
    )
    
    # Example 2: Image + Text
    # ticket_2 = Ticket(
    #     ticket_id="T-002",
    #     text="Look at what I found at the gate.",
    #     image_path="/path/to/image.png"
    # )
    
    result = agent.process(ticket_1)
    print("\n=== Agent Result ===")
    print(result.model_dump_json(indent=2))