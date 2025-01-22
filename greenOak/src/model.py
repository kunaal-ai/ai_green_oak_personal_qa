import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Dict, List, Optional
import json
import logging
from pathlib import Path

class TestCaseGenerator(nn.Module):
    def __init__(self, model_name: str = "t5-base", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Special tokens for test case generation
        special_tokens = {
            'additional_special_tokens': [
                '[FEATURE]', '[DOMAIN]', '[TYPE]', '[SCENARIO]',
                '[STEPS]', '[EXPECTED]', '[DATA]', '[REQUIREMENTS]'
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def format_input(self, feature_description: str, feature_type: str, test_type: str, domain: Optional[str] = None) -> str:
        """Format input text for the model"""
        input_text = f"Generate {test_type} test case for {feature_type}:"
        input_text += f" {feature_description}"
        if domain:
            input_text += f" Domain: {domain}"
        return input_text

    def format_output(self, output_text: str) -> Optional[Dict]:
        """Format model output into structured test case"""
        try:
            # Initialize default structure
            test_case = {
                "scenario": "",
                "steps": [],
                "expected_results": [],
                "test_data": {}
            }
            
            # Extract sections using markers
            sections = {
                "scenario": "[SCENARIO]",
                "steps": "[STEPS]",
                "expected": "[EXPECTED]",
                "data": "[DATA]"
            }
            
            current_section = None
            for line in output_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line is a section marker
                found_section = False
                for section, marker in sections.items():
                    if marker in line:
                        current_section = section
                        line = line.replace(marker, "").strip()
                        found_section = True
                        break
                
                if not found_section and current_section:
                    # Process line based on current section
                    if current_section == "scenario":
                        test_case["scenario"] = line
                    elif current_section == "steps":
                        if line.startswith("- "):
                            line = line[2:]
                        test_case["steps"].append(line)
                    elif current_section == "expected":
                        if line.startswith("- "):
                            line = line[2:]
                        test_case["expected_results"].append(line)
                    elif current_section == "data":
                        try:
                            test_case["test_data"] = json.loads(line)
                        except:
                            test_case["test_data"] = {"value": line}

            # Validate test case has required fields
            if not test_case["scenario"] and not test_case["steps"]:
                return None
                
            return test_case
            
        except Exception as e:
            logging.error(f"Error formatting output: {str(e)}")
            return None

    def generate(self,
                feature_description: str,
                feature_type: str,
                test_type: str,
                domain: Optional[str] = None,
                num_return_sequences: int = 1,
                max_length: int = 512) -> List[Dict]:
        """Generate test cases"""
        try:
            # Format input
            input_text = self.format_input(feature_description, feature_type, test_type, domain)
            
            # Tokenize input
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(self.device)
            
            # Generate outputs
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                num_beams=5,
                no_repeat_ngram_size=3,
                temperature=0.8,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id
            )
            
            # Decode and format outputs
            decoded_outputs = []
            for output in outputs:
                decoded_text = self.tokenizer.decode(output, skip_special_tokens=True)
                formatted_output = self.format_output(decoded_text)
                if formatted_output:  # Only add non-empty outputs
                    decoded_outputs.append(formatted_output)
            
            if not decoded_outputs:
                # If no valid outputs, create a basic test case
                return [{
                    "scenario": f"Test {feature_type} functionality",
                    "steps": [
                        f"Initialize {feature_type} system",
                        f"Configure {test_type} test parameters",
                        f"Execute {feature_type} operation",
                        "Verify results"
                    ],
                    "expected_results": [
                        f"{feature_type} system should initialize correctly",
                        "Operation should complete successfully",
                        "Results should match expected behavior"
                    ],
                    "test_data": {
                        "type": feature_type,
                        "test_category": test_type,
                        "domain": domain if domain else "general"
                    }
                }]
            
            return decoded_outputs
            
        except Exception as e:
            logging.error(f"Error in generate: {str(e)}")
            return [{
                "scenario": f"Test {feature_type} functionality",
                "steps": [
                    f"Initialize {feature_type} system",
                    "Perform basic operation",
                    "Verify results"
                ],
                "expected_results": [
                    "System should respond correctly",
                    "Operation should complete successfully"
                ],
                "test_data": {}
            }]

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        outputs = self.model(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            labels=batch["labels"].to(self.device)
        )
        
        return {
            "loss": outputs.loss.item(),
            "logits": outputs.logits
        }

    def save_pretrained(self, save_path: str):
        """Save the model and tokenizer"""
        # Create directory if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, path: str):
        """Load the model and tokenizer"""
        self.model = T5ForConditionalGeneration.from_pretrained(path).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(path)
