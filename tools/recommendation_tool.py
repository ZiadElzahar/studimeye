import torch
from transformers import pipeline

class RecommendationTool:
    def __init__(self):
        model_name = "Qwen/Qwen2.5-1.5B-Instruct" 
        self.pipe = pipeline("text-generation", model=model_name, device=0 if torch.cuda.is_available() else -1)

    def create_prompt(self, text: str) -> str:
        prompt = """Extract the core inquiry from the support log. Focus strictly on the issue type, the subject, and the location. Remove all filler text about who submitted it. Keep it under 10 words.

Examples:
Input: family group traveling together with children submitted a accessibility request about airport_arrival in gate a.
Output: accessibility request in gate A

Input: family group traveling together with children submitted a emergency report about restroom in food court.
Output: emergency report about restroom in food court

Input: {input_text}
Output:"""
        return prompt.format(input_text=text)

    def process(self, text: str) -> str:
        """Public method called by orchestrator. Returns summary string."""
        prompt = self.create_prompt(text)
        results = self.pipe(
            prompt,
            max_new_tokens=20,
            do_sample=False,
            return_full_text=False
        )
        summary = results[0]['generated_text'].strip().split('\n')[0] 
        return summary

if __name__ == "__main__":
    tool = RecommendationTool()
    # print(tool.process("Test input for core inquiry extraction."))
    pass