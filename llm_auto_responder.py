#!/usr/bin/env python3
"""
LLM Auto-Responder for Cross Rulings Training

This module provides automated responses to classification questions using
the full cross ruling context to simulate knowledgeable user interactions.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the api directory to the Python path
script_dir = Path(__file__).parent
api_dir = script_dir / "api"
sys.path.insert(0, str(api_dir))

from api.llm_client import LLMClient

class LLMAutoResponder:
    def __init__(self, engine_name: str = "groq", debug: bool = False):
        self.engine_name = engine_name
        self.debug = debug
        self.setup_logging()
        self.llm_client = LLMClient()
        
    def setup_logging(self):
        """Configure logging"""
        level = logging.DEBUG if self.debug else logging.INFO
        self.logger = logging.getLogger(__name__)
        
    def generate_response(self, question: Dict, cross_ruling: Dict, conversation_history: List[Dict] = None) -> str:
        """
        Generate an appropriate response to a classification question using cross ruling context
        
        Args:
            question: The clarification question from the classification engine
            cross_ruling: The full cross ruling data with complete product description
            conversation_history: Previous Q&A pairs in this classification session
            
        Returns:
            str: The generated response that should answer the question appropriately
        """
        
        # Extract question details
        if hasattr(question, 'question_text'):
            question_text = question.question_text
            options = getattr(question, 'options', [])
        else:
            question_text = question.get("question_text", "")
            options = question.get("options", [])
        
        # Get full product context from cross ruling
        full_description = cross_ruling.get("full_description", "")
        product_description = cross_ruling.get("short_product_description", "")
        hts_code = cross_ruling.get("hts_code", "")
        reasoning = cross_ruling.get("reasoning", "")
        
        # CRITICAL VALIDATION: Ensure we have the essential product information
        if not full_description:
            self.logger.error("CRITICAL: No full_description in cross ruling data!")
            self.logger.error(f"Available fields: {list(cross_ruling.keys())}")
            self.logger.error("This will cause classification failure!")
        
        # Combine full description with reasoning for complete context
        combined_context = f"{full_description}"
        if reasoning:
            combined_context += f"\n\nCLASSIFICATION REASONING: {reasoning}"
        
        # ENHANCED LOGGING: Log the complete context being used
        self.logger.info("="*80)
        self.logger.info("AUTO-RESPONDER PROCESSING NEW QUESTION")
        self.logger.info("="*80)
        self.logger.info(f"SHORT DESCRIPTION: {product_description}")
        self.logger.info(f"FULL DESCRIPTION: {full_description}")
        self.logger.info(f"EXPECTED HTS CODE: {hts_code}")
        self.logger.info(f"EXPECTED REASONING: {reasoning}")
        self.logger.info(f"QUESTION ASKED: {question_text}")
        
        # Debug verification
        self.logger.info(f"USING PRODUCT DATA: {full_description[:200]}...")
        if not full_description.strip():
            self.logger.error("❌ EMPTY PRODUCT DESCRIPTION - THIS WILL FAIL!")
        else:
            self.logger.info("✓ Product description loaded successfully")
        
        # Build options context
        options_context = ""
        if options:
            options_context = "Available options:\n"
            for i, option in enumerate(options, 1):
                if isinstance(option, dict):
                    option_text = option.get('text', str(option))
                else:
                    option_text = str(option)
                options_context += f"{i}. {option_text}\n"
            
            self.logger.info("AVAILABLE OPTIONS:")
            for i, option in enumerate(options, 1):
                if isinstance(option, dict):
                    option_text = option.get('text', str(option))
                else:
                    option_text = str(option)
                self.logger.info(f"  {i}. {option_text}")
        
        # Log conversation history if provided
        if conversation_history:
            self.logger.info("CONVERSATION HISTORY:")
            for i, qa in enumerate(conversation_history, 1):
                self.logger.info(f"  Q{i}: {qa.get('question', 'N/A')}")
                self.logger.info(f"  A{i}: {qa.get('answer', 'N/A')}")
        
        # Create simple prompt for JSON response
        if options:
            # Multiple choice question - request JSON with option number
            prompt = f"""You are answering based on OFFICIAL CBP RULING DATA. This is the authoritative product specification that must be followed exactly.

OFFICIAL PRODUCT SPECIFICATION: {combined_context}

CLASSIFICATION QUESTION: {question_text}

{options_context}

CRITICAL: Answer based ONLY on the official product specification above, not general knowledge about similar products. The specification contains the exact technical details needed for classification.

Required JSON format: {{"selected_option": "1"}}"""
            
            self.logger.info("SENDING PROMPT TO LLM:")
            self.logger.info("-" * 60)
            self.logger.info(prompt)
            self.logger.info("-" * 60)
            
            try:
                # Use OpenAI o3 with JSON response
                response = self.llm_client.send_openai_request(
                    prompt=prompt,
                    requires_json=True,
                    temperature=0
                )
                
                self.logger.info("RAW LLM RESPONSE:")
                self.logger.info(response)
                
                # Parse JSON response
                import json
                response_data = json.loads(response)
                selected_option = response_data.get("selected_option", "1")
                
                self.logger.info(f"PARSED SELECTED OPTION: {selected_option}")
                
                # Enhanced validation and logging
                try:
                    option_num = int(selected_option)
                    if 1 <= option_num <= len(options):
                        if isinstance(options[option_num - 1], dict):
                            selected_text = options[option_num - 1].get('text', str(options[option_num - 1]))
                        else:
                            selected_text = str(options[option_num - 1])
                        self.logger.info(f"VALID SELECTION - Option {option_num}: {selected_text}")
                    else:
                        self.logger.warning(f"INVALID OPTION NUMBER: {option_num} (valid range: 1-{len(options)})")
                except ValueError:
                    self.logger.warning(f"NON-NUMERIC OPTION: {selected_option}")
                
                self.logger.info("="*80)
                return selected_option
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON PARSING ERROR: {e}")
                self.logger.error(f"Raw response was: {response}")
                self.logger.info("FALLBACK: Using option 1")
                self.logger.info("="*80)
                return "1"
            except Exception as e:
                self.logger.error(f"LLM REQUEST ERROR: {e}")
                self.logger.info("FALLBACK: Using option 1")
                self.logger.info("="*80)
                return "1"
        else:
            # Free text question
            prompt = f"""You are answering based on OFFICIAL CBP RULING DATA. Answer the question using only the authoritative product specification provided.

OFFICIAL PRODUCT SPECIFICATION: {combined_context}

QUESTION: {question_text}

CRITICAL: Answer based ONLY on the official product specification above, not general knowledge. Provide a concise, accurate answer based on the exact product details given."""
            
            self.logger.info("SENDING TEXT PROMPT TO LLM:")
            self.logger.info("-" * 60)
            self.logger.info(prompt)
            self.logger.info("-" * 60)
            
            try:
                response = self.llm_client.send_openai_request(
                    prompt=prompt,
                    requires_json=False,
                    temperature=0.3
                )
                
                self.logger.info("RAW LLM RESPONSE:")
                self.logger.info(response)
                self.logger.info(f"✓ FINAL TEXT RESPONSE: {response.strip()}")
                self.logger.info("="*80)
                return response.strip()
                
            except Exception as e:
                self.logger.error(f"❌ LLM REQUEST ERROR: {e}")
                fallback_response = "Please provide more specific details."
                self.logger.info(f"FALLBACK: {fallback_response}")
                self.logger.info("="*80)
                return fallback_response
    
    def interactive_classify_with_auto_response(self, hts_tree, product_description: str, cross_ruling: Dict, max_questions: int = 5, hypothesis_count: int = None) -> Dict:
        """
        Run interactive classification with automatic LLM responses
        
        Args:
            hts_tree: The HTS classification tree instance
            product_description: The abbreviated product description for classification
            cross_ruling: The full cross ruling data for generating responses
            max_questions: Maximum number of questions to ask
            hypothesis_count: Beam width for classification (default: from TREERL_BEAM_SIZE env)
            
        Returns:
            Dict: The final classification result with conversation history
        """
        import os
        
        # Use env var if not specified
        if hypothesis_count is None:
            hypothesis_count = int(os.environ.get("TREERL_BEAM_SIZE", "4"))
        
        self.logger.info(f"Starting auto-interactive classification for: {product_description}")
        self.logger.info(f"Using beam_size={hypothesis_count}, max_questions={max_questions}")
        
        # Start classification
        result = hts_tree.start_classification(
            product=product_description,
            interactive=True,
            max_questions=max_questions,
            use_multi_hypothesis=True,
            hypothesis_count=hypothesis_count
        )
        
        conversation_history = []
        question_count = 0
        
        # Handle interactive loop with auto-responses
        while not result.get("final", True) and result.get("clarification_question") and question_count < max_questions:
            question = result.get("clarification_question")
            question_count += 1
            
            # Extract question text for logging
            if hasattr(question, 'question_text'):
                question_text = question.question_text
            else:
                question_text = question.get("question_text", "")
            
            self.logger.info(f"Question {question_count}: {question_text}")
            
            # Generate automatic response
            response = self.generate_response(question, cross_ruling, conversation_history)
            
            # Log the Q&A pair
            qa_pair = {
                "question": question_text,
                "answer": response,
                "question_number": question_count
            }
            conversation_history.append(qa_pair)
            
            self.logger.info(f"Auto-response {question_count}: {response}")
            
            # Continue classification with the response
            try:
                result = hts_tree.continue_classification(
                    state=result.get("state", {}),
                    answer=response,
                    interactive=True,
                    max_questions=max_questions
                )
            except Exception as e:
                self.logger.error(f"Error continuing classification: {e}")
                break
        
        # Add conversation history to result
        if conversation_history:
            result["auto_conversation"] = conversation_history
            result["total_questions"] = len(conversation_history)
        
        return result

def test_auto_responder():
    """Test the auto-responder with sample data"""
    
    # Sample cross ruling data
    sample_cross_ruling = {
        "product_description": "AIO fingerprint reader",
        "full_description": "The Grabba X-Series biometric scanning device comprised of fingerprint reader, passport reader, facial recognition scanner, barcode reader, RFID and Smart Card reader within rugged enclosure. USB connector designed to connect to compatible Android smartphone for law enforcement, retail, security, border protection, emergency response, travel, warehousing to capture biometric and biographic data.",
        "hts_code": "8543.70.9860",
        "reasoning": "Not principally used in ADP system, nor connectable to CPU directly or through other ADP units."
    }
    
    # Sample question
    sample_question = {
        "question_text": "Does your AIO fingerprint reader connect directly to a computer's central processing unit, or does it require connection through another device like a smartphone?",
        "options": [
            "Connects directly to CPU for data processing",
            "Requires connection through smartphone or other device",
            "Can work both ways depending on configuration"
        ]
    }
    
    responder = LLMAutoResponder(debug=True)
    response = responder.generate_response(sample_question, sample_cross_ruling)
    
    print("Sample Question:", sample_question["question_text"])
    print("Generated Response:", response)

if __name__ == "__main__":
    test_auto_responder()
