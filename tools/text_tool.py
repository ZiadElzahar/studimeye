import os
import json
import re
import time
from typing import Any, Dict, List

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

TAXONOMY: Dict[str, List[str]] = {
    "categories": ["ticketing", "transportation", "security", "medical", "lost and found", 
                   "stadium facilities", "food and beverage", "accessibility", "technology", 
                   "staff", "fan zone", "hotel", "airport", "visa and travel", "tourism", "crowd management"],
    "personas": ["tourist", "local fan", "family", "student", "elderly", "disabled fan", 
                 "vip", "journalist", "volunteer", "security officer", "stadium staff", 
                 "international visitor", "football ultra"],
    "interactions": ["Complaint", "Incident Report", "Assistance Request", "Information Inquiry", 
                     "Feedback", "Suggestion", "Appreciation", "Emergency Report", "Technical Issue", 
                     "Accessibility Request", "Lost & Found", "Travel Support", "Accommodation Support", "Payment Issue"],
    "departments": ["Security", "Medical", "Transportation", "Ticketing", "Customer Service", 
                    "IT Support", "Lost & Found", "Volunteers", "Facility Management", "Cleaning", 
                    "Food Services", "Police", "Emergency Response", "Hotel Support", "Tourism Office"],
    "severities": ["Low", "Medium", "High", "Critical"]
}

class TextTool:
    def __init__(self) -> None:
        print("Loading Local SLM (Phi-3-mini) for Generative Triaging...")
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        config = AutoConfig.from_pretrained(model_id,trust_remote_code=True)
        config.use_cache = False 
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, config=config, device_map="auto", torch_dtype=torch.float16,trust_remote_code=True
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        print("SLM Agent is ready for Classification, NER, and Suggestion Generation!")

    def build_prompt(self, ticket_text: str) -> str:
        system_prompt = f"""You are an expert customer service AI agent for a major event. 
Your task is to analyze the user's ticket/complaint and perform THREE tasks:
1. Extract specific metadata (Classification).
2. Extract the physical location (NER).
3. Provide 2-3 actionable recommendations for the support team (Generative Action).

You MUST choose your classification answers STRICTLY from the following lists:
- categories: {', '.join(TAXONOMY['categories'])}
- personas: {', '.join(TAXONOMY['personas'])}
- interactions: {', '.join(TAXONOMY['interactions'])}
- departments: {', '.join(TAXONOMY['departments'])}
- severities: {', '.join(TAXONOMY['severities'])}

For "location", extract the physical location mentioned in the text (e.g., "Gate D", "Terminal 1", "Food Court"). If none is mentioned, output "Not Found".
For "suggestions", provide a JSON array of 2 to 3 short, practical steps the customer service team should take immediately.

OUTPUT FORMAT:
You must reply with a valid JSON object ONLY. Do not write any explanations, greetings, or other text.
Example format:
{{
    "category": "security",
    "persona": "tourist",
    "interaction": "Incident Report",
    "department": "Security",
    "severity": "High",
    "location": "Gate D",
    "suggestions": [
        "Dispatch nearest security patrol to Gate D immediately.",
        "Review CCTV footage for suspicious activities.",
        "Assist the tourist in contacting local authorities."
    ]
}}"""
        user_prompt = f"Analyze the following ticket, extract the metadata, and provide suggestions as JSON:\n\nTicket: '{ticket_text}'"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def parse_json(self, raw_output: str) -> Dict[str, Any]:
        try:
            match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                return {"error": "No JSON object found in output.", "raw": raw_output}
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON.", "raw": raw_output}

    def process(self, text: str) -> Dict[str, Any]:
        prompt = self.build_prompt(text)
        outputs = self.pipe(prompt, max_new_tokens=300, temperature=0.1, do_sample=True, return_full_text=False)
        parsed_data = self.parse_json(outputs[0]['generated_text'])
        if "severity" in parsed_data:
            priority_map = {"Low": "P4", "Medium": "P3", "High": "P2", "Critical": "P1"}
            parsed_data["priority"] = priority_map.get(parsed_data["severity"], "P3")
        return parsed_data

# ============================================================
# ORIGINAL TESTS (Preserved exactly as requested)
# ============================================================
if __name__ == "__main__":
    agent = TextTool()
    test_cases = [
        "Someone just stole my backpack while I was waiting near Gate D. My passport and wallet are inside and my flight leaves in 2 hours!",
        "The elevator leading to the VIP Lounge in Sector 4 is broken. I have a wheelchair and cannot take the stairs.",
        "المقاعد في المنطقة المخصصة للعائلات في المدرج الشرقي متسخة جداً ولم يتم تنظيفها، نحن منزعجون جداً."
    ]
    print("\n" + "="*70)
    for i, text in enumerate(test_cases, 1):
        print(f"--- Processing Ticket #{i} ---")
        print(f"User Ticket: {text}\n")
        start_time = time.time()
        output = agent.process(text)
        elapsed_time = time.time() - start_time
        if "error" in output:
            print("ERROR IN EXTRACTION:")
            print(output["raw"])
        else:
            for k, v in output.items():
                if k != "suggestions":
                    print(f"{k.capitalize():<15}: {v}")
            print(f"\n{'Suggestions':<15}:")
            if "suggestions" in output and isinstance(output["suggestions"], list):
                for idx, suggestion in enumerate(output["suggestions"], 1):
                    print(f"  {idx}. {suggestion}")
            else:
                print("  No valid suggestions array found.")
        print(f"\n[Processing Time: {elapsed_time:.2f} seconds]")
        print("="*70)