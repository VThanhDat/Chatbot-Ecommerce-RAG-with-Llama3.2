import os
import torch
from dotenv import load_dotenv
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = Path(r"F:/NamTu/HK2/NLP/Chatbot-Ecommerce-RAG-with-Llama3.2/Llama-3.2-3B-Instruct")

class EcommerceChatbotManager:
    def __init__(self, fine_tune_model, device=None):
        self.model_path = model_path
        self.fine_tune_model = fine_tune_model
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.system_instruction = "You are a top-rated customer service agent for an e-commerce company."

        if not self.model_path.exists():
            raise FileNotFoundError(f"⚠️ Model path không tồn tại: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        base_model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="cpu")
        self.model = PeftModel.from_pretrained(base_model, self.fine_tune_model).to(self.device)
        
        print("✅ Model loaded successfully!")

    def chat(self, prompt, instruction=None):
        instruction = instruction if instruction else self.system_instruction

        chat = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ]

        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            inputs = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.device)
        else:
            formatted_prompt = f"System: {instruction}\nUser: {prompt}\nAssistant:"
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
        # Look for both "Assistant:" and "assistant" patterns
        assistant_markers = ["Assistant:", "assistant"]
        for marker in assistant_markers:
            assistant_index = response.find(marker)
            if assistant_index != -1:
                response = response[assistant_index + len(marker):].strip()
                break
        
        # Remove any remaining role markers
        unwanted_prefixes = ["system", "user", "assistant"]
        for prefix in unwanted_prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Replace template variables
        replacements = {
            "{{Website URL}}": "www.yourstore.com",
            "{{Customer Support Phone Number}}": "1-800-123-4567",
            "{{Newsletter Category}}": "Winter Fashion"
        }
        
        for placeholder, value in replacements.items():
            response = response.replace(placeholder, value)
        
        return response
