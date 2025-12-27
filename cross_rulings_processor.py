#!/usr/bin/env python3
"""
Cross Rulings Training Dataset Processor

Processes cross rulings dataset to generate training data by:
1. Loading cross rulings from JSON file
2. Injecting full cross ruling info into system prompt
3. Starting classification with abbreviated product description
4. Collecting training data with proper confidence scoring
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

from api.system_prompts_updated import CORE_PROMPT
from api.training_data_collector import training_collector
from llm_auto_responder import LLMAutoResponder

class CrossRulingsProcessor:
    def __init__(self, engine_name: str = "groq", debug: bool = False, use_auto_responder: bool = True):
        self.engine_name = engine_name
        self.debug = debug
        self.use_auto_responder = use_auto_responder
        self.setup_logging()
        self.load_engine()
        
        # Initialize auto-responder if enabled
        if self.use_auto_responder:
            self.auto_responder = LLMAutoResponder(engine_name=engine_name, debug=debug)
        else:
            self.auto_responder = None
        
    def setup_logging(self):
        """Configure logging - same as manual CLI classifier"""
        if self.debug:
            level = logging.DEBUG
        else:
            level = logging.INFO
            
        # Configure file logging (always detailed)
        file_handler = logging.FileHandler('cross_rulings_training.log')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Configure console logging (respects debug mode)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Allow all levels to file
        root_logger.handlers.clear()  # Clear any existing handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Always suppress HTTP and connection noise - we don't need to see this
        logging.getLogger('httpx').setLevel(logging.ERROR)
        logging.getLogger('httpcore').setLevel(logging.ERROR)
        logging.getLogger('httpcore.http11').setLevel(logging.ERROR)
        logging.getLogger('httpcore.connection').setLevel(logging.ERROR)
        logging.getLogger('urllib3').setLevel(logging.ERROR)
        logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)
        logging.getLogger('google.auth').setLevel(logging.ERROR)
        logging.getLogger('google.auth._default').setLevel(logging.ERROR)
        
        # Suppress OpenAI library debug logs
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('openai._base_client').setLevel(logging.WARNING)
        logging.getLogger('openai.resources').setLevel(logging.WARNING)
        logging.getLogger('openai.lib._parsing').setLevel(logging.WARNING)
        logging.getLogger('openai._client').setLevel(logging.WARNING)
        
        # Suppress Groq library debug logs to prevent Unicode encoding errors
        logging.getLogger('groq').setLevel(logging.WARNING)
        logging.getLogger('groq.resources').setLevel(logging.WARNING)
        logging.getLogger('groq.resources.chat').setLevel(logging.WARNING)
        logging.getLogger('groq.resources.chat.completions').setLevel(logging.WARNING)
        logging.getLogger('groq._base_client').setLevel(logging.WARNING)
        
        # Keep GenAI logs but suppress the most verbose ones
        if not self.debug:
            # In normal mode, keep important GenAI logs but suppress verbose ones
            logging.getLogger('google_genai.models').setLevel(logging.WARNING)
        else:
            # In debug mode, show GenAI logs but not the most verbose ones
            logging.getLogger('google_genai.models').setLevel(logging.INFO)
            
        self.logger = logging.getLogger(__name__)
        
    def load_engine(self):
        """Load the specified classification engine"""
        available_engines = {
            "cerebras": "cerebras_tree_engine",
            "groq": "groq_tree_engine", 
            "tree": "tree_engine"
        }
        
        if self.engine_name not in available_engines:
            available = ", ".join(available_engines.keys())
            raise ValueError(f"Unknown engine '{self.engine_name}'. Available: {available}")
        
        module_name = available_engines[self.engine_name]
        
        try:
            module = __import__(f"api.{module_name}", fromlist=[module_name])
            if not hasattr(module, 'HTSTree'):
                raise ImportError(f"Engine '{self.engine_name}' does not have an HTSTree class")
            
            self.HTSTree = module.HTSTree
            self.logger.info(f"Using {self.engine_name} engine ({module_name})")
            
        except ImportError as e:
            self.logger.error(f"Error importing {self.engine_name} engine: {e}")
            raise
            
    def load_cross_rulings(self, file_path: str) -> List[Dict]:
        """Load cross rulings from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cross_rulings = json.load(f)
            
            self.logger.info(f"Loaded {len(cross_rulings)} cross rulings from {file_path}")
            return cross_rulings
            
        except Exception as e:
            self.logger.error(f"Error loading cross rulings: {e}")
            raise
            
    def inject_cross_ruling_into_prompt(self, cross_ruling: Dict) -> str:
        """Inject cross ruling information into the system prompt"""
        # Use the actual cross ruling data from the dataset
        full_description = cross_ruling.get("full_cross_ruling", "")
        hts_code = cross_ruling.get("hts_code", "")
        reasoning = cross_ruling.get("reasoning", "")
        
        cross_ruling_text = f"""
RE: The tariff classification of a product

The merchandise under consideration is described as follows:

{full_description}

Classification Analysis:
{reasoning}

The applicable classification is {hts_code}.

This represents a customs ruling demonstrating proper classification methodology and reasoning.
"""
        
        # Replace the cross ruling section in the system prompt
        base_prompt = CORE_PROMPT
        
        # Find the "Cross Ruling:" section and replace it
        training_section_start = base_prompt.find("## Training")
        if training_section_start == -1:
            # If no training section found, append it
            modified_prompt = base_prompt + f"\n\n## Training\nNote we are now in training dataset collection mode. What this means is that I will provide you will CROSS ruling description pair and then you must ensure classification obtains this result.\nYOU MUST ENSURE THAT THE SCORING FOR ALL CANDIDATES IS 100% CONCORDANT WITH THE GUIDELINES ABOVE. DO NOT RUSH MATCHING THE CONFIDENCE AS THIS IS THE WHOLE POINT. YOU ASSIGN ACCURATE CONFIDENCE, SELECTION LABELS ECT AS YOU ARE THE TEACHER MODEL GENERATING THE TRAINING DATASET FOR THE WEAKER MODEL WITH NO INSTRUCTIONS. MOST IMPORTANT YOU MUST RESPECT THE CONFIDENCE THRESHOLD FOR ASKING QUESTIONS AS WHILE I AM PROVIDING YOU FULL PRODUCT CONTEXT HERE THE PROMPT WILL NOT HAVE FULL CONTEXT. THUS WHEN INFORMATION IN THE PROMPT IS NOT AVALIABLE TO ANSWER THE CLASSIFICATION OPTIONS YOU MUST HAVE LOWER CONFIDENCE TO ASK A QUESTION. IF YOU DO NOT ASK QUESTIONS AND THE CROSS RULING IS FOR A REALLY SPECIFIC PRODUCT AND I PROVIDE A SHORTENED DESCRIPTION YOU WILL TEACH THE STUDENT MODEL TO OVERFIT TO THE FULL CROSS RULINGS DESCRIPTION WHEN THATS NOT WHAT THE USER INPUTED. YOU CANNOT KNOW WITHOUT ASKING.\nDO NOT MENTION THIS CROSS RULING INFORMATION IN THE REASONING, WE ARE TRYING TO TEACH HOW TO PREFORM ACCURATE CLASSIFICATION.\n\nCross Ruling:\n{cross_ruling_text}"
        else:
            # Replace existing cross ruling
            cross_ruling_start = base_prompt.find("Cross Ruling:")
            if cross_ruling_start != -1:
                # Keep everything up to "Cross Ruling:" and replace the rest
                modified_prompt = base_prompt[:cross_ruling_start] + f"Cross Ruling:\n{cross_ruling_text}"
            else:
                # Add cross ruling to existing training section
                modified_prompt = base_prompt + f"\n\nCross Ruling:\n{cross_ruling_text}"
        
        return modified_prompt
        
    def process_single_cross_ruling(self, cross_ruling: Dict, index: int = 0) -> Dict:
        """Process a single cross ruling to generate training data"""
        product_description = cross_ruling["short_product_description"]
        expected_code = cross_ruling["hts_code"]
        
        self.logger.info(f"Processing cross ruling {index + 1}: {product_description}")
        
        try:
            # Enable training data collection
            os.environ["COLLECT_TRAINING_DATA"] = "true"
            
            # Start training session
            session_id = f"cross_ruling_{index + 1}"
            training_collector.start_session(product_description, session_id)
            
            # Create HTS tree with modified system prompt
            modified_prompt = self.inject_cross_ruling_into_prompt(cross_ruling)
            hts_tree = self.HTSTree()
            
            # Enable debug logging on the classification engine if debug mode is on
            if self.debug:
                # Set environment variable that some engines check
                os.environ["LOG_PROMPTS"] = "true"
                os.environ["DEBUG"] = "true"
                
                # Configure the engine's logger to debug level
                engine_logger = logging.getLogger(hts_tree.__class__.__module__)
                engine_logger.setLevel(logging.DEBUG)
                
                # Also set the classification engine logger
                classification_logger = logging.getLogger('api.classification_engine')
                classification_logger.setLevel(logging.DEBUG)
                
                # Set groq engine logger if applicable
                groq_logger = logging.getLogger('api.groq_tree_engine')
                groq_logger.setLevel(logging.DEBUG)
                
                self.logger.info("DEBUG MODE: Enabled debug logging for classification engine")
            
            # Load HTS data
            hts_data_file = script_dir / "api" / "hts_data.json"
            with open(hts_data_file, "r", encoding="utf-8") as f:
                hts_data = json.load(f)
            hts_tree.build_from_json(hts_data)
            
            # Override the system prompt in the tree
            if hasattr(hts_tree, 'system_prompt'):
                hts_tree.system_prompt = modified_prompt
            elif hasattr(hts_tree, 'llm_client') and hasattr(hts_tree.llm_client, 'system_prompt'):
                hts_tree.llm_client.system_prompt = modified_prompt
            
            # Start classification with abbreviated description
            self.logger.info(f"Starting classification for: {product_description}")
            result = hts_tree.start_classification(
                product=product_description,
                interactive=True,  # Enable interactive mode for questions
                max_questions=5,
                use_multi_hypothesis=True,
                hypothesis_count=3
            )
            
            # Check if this is final or needs interaction
            classification_success = False
            conversation_log = []
            
            # Handle interactive classification loop
            question_count = 0
            max_questions = 5
            
            self.logger.info("=" * 60)
            self.logger.info("STARTING INTERACTIVE CLASSIFICATION LOOP")
            self.logger.info("=" * 60)
            self.logger.info(f"Product: '{product_description}' | Expected: {expected_code}")
            
            while not result.get("final", True) and result.get("clarification_question") and question_count < max_questions:
                question = result.get("clarification_question")
                question_count += 1
                
                # Log the question
                if hasattr(question, 'question_text'):
                    question_text = question.question_text
                    options = getattr(question, 'options', [])
                else:
                    question_text = question.get("question_text", "")
                    options = question.get("options", [])
                
                self.logger.info("-" * 60)
                self.logger.info(f"QUESTION {question_count} OF {max_questions}")
                self.logger.info("-" * 60)
                self.logger.info(f"QUESTION: {question_text}")
                
                if options:
                    self.logger.info("AVAILABLE OPTIONS:")
                    for i, option in enumerate(options, 1):
                        if isinstance(option, dict):
                            option_text = option.get('text', str(option))
                        else:
                            option_text = str(option)
                        self.logger.info(f"   {i}. {option_text}")
                
                if self.use_auto_responder and self.auto_responder:
                    # Use LLM auto-responder
                    try:
                        self.logger.info("CALLING AUTO-RESPONDER...")
                        response = self.auto_responder.generate_response(question, cross_ruling, conversation_log)
                        
                        # Log the Q&A pair
                        qa_pair = {
                            "question": question_text,
                            "answer": response,
                            "question_number": question_count
                        }
                        conversation_log.append(qa_pair)
                        
                        self.logger.info(f"AUTO-RESPONSE {question_count}: '{response}'")
                        
                        # Validate the response
                        if options:
                            try:
                                response_num = int(response)
                                if 1 <= response_num <= len(options):
                                    if isinstance(options[response_num - 1], dict):
                                        selected_text = options[response_num - 1].get('text', str(options[response_num - 1]))
                                    else:
                                        selected_text = str(options[response_num - 1])
                                    self.logger.info(f"SELECTED: Option {response_num} = '{selected_text}'")
                                else:
                                    self.logger.warning(f"INVALID RESPONSE: {response} (valid: 1-{len(options)})")
                            except ValueError:
                                self.logger.warning(f"NON-NUMERIC RESPONSE: '{response}'")
                        
                        # Continue classification with the response
                        self.logger.info("CONTINUING CLASSIFICATION WITH RESPONSE...")
                        result = hts_tree.continue_classification(
                            state=result.get("state", {}),
                            answer=response,
                            interactive=True,
                            max_questions=max_questions
                        )
                        
                        # Log the updated result
                        if result.get("final", True):
                            final_code = None
                            if "classification" in result:
                                final_code = result["classification"].get("code")
                            else:
                                final_code = result.get("final_code")
                            self.logger.info(f"CLASSIFICATION COMPLETED: {final_code}")
                        else:
                            self.logger.info("CLASSIFICATION CONTINUES - MORE QUESTIONS NEEDED")
                        
                    except Exception as e:
                        self.logger.error(f"ERROR IN AUTO-RESPONDER: {e}")
                        import traceback
                        self.logger.error(f"TRACEBACK: {traceback.format_exc()}")
                        break
                else:
                    # Manual intervention required
                    self.logger.warning("MANUAL MODE: Auto-responder disabled - breaking")
                    break
            
            # Check final result
            if result.get("final", True):
                final_code = None
                if "classification" in result:
                    final_code = result["classification"].get("code")
                else:
                    final_code = result.get("final_code")
                
                # Normalize codes for comparison (remove all dots to handle formatting differences)
                def normalize_code(code):
                    if code:
                        return str(code).replace(".", "")
                    return ""
                
                normalized_final = normalize_code(final_code)
                normalized_expected = normalize_code(expected_code)
                
                # Always log the detailed comparison for debugging
                self.logger.info(f"Code comparison - Final: '{final_code}' | Expected: '{expected_code}'")
                self.logger.info(f"Normalized - Final: '{normalized_final}' | Expected: '{normalized_expected}'")
                
                if normalized_final == normalized_expected:
                    classification_success = True
                    self.logger.info(f"✓ SUCCESS: Classification matches expected code")
                else:
                    self.logger.warning(f"✗ MISMATCH: Classification does not match expected code")
            
            # End training session
            training_collector.end_session(result, classification_success)
            
            return {
                "success": classification_success,
                "result": result,
                "expected_code": expected_code,
                "conversation_log": conversation_log
            }
            
        except Exception as e:
            self.logger.error(f"Error processing cross ruling {index + 1}: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            
            # End session with failure
            training_collector.end_session({}, False)
            
            return {
                "success": False,
                "error": str(e),
                "expected_code": expected_code
            }
            
    def process_cross_rulings_dataset(self, file_path: str, max_count: Optional[int] = None) -> Dict:
        """Process the entire cross rulings dataset"""
        cross_rulings = self.load_cross_rulings(file_path)
        
        if max_count:
            cross_rulings = cross_rulings[:max_count]
            
        total_count = len(cross_rulings)
        successful_count = 0
        results = []
        
        self.logger.info(f"Processing {total_count} cross rulings...")
        
        for i, cross_ruling in enumerate(cross_rulings):
            result = self.process_single_cross_ruling(cross_ruling, i)
            results.append(result)
            
            if result["success"]:
                successful_count += 1
                
        success_rate = (successful_count / total_count) * 100 if total_count > 0 else 0
        
        summary = {
            "total_processed": total_count,
            "successful": successful_count,
            "success_rate": success_rate,
            "results": results
        }
        
        self.logger.info(f"Processing complete: {successful_count}/{total_count} successful ({success_rate:.1f}%)")
        
        return summary

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process cross rulings dataset for training data generation")
    parser.add_argument("--cross-rulings-file", default="cross_rulings_dataset.json", 
                       help="Path to cross rulings JSON file")
    parser.add_argument("--engine", choices=["cerebras", "groq", "tree"], default="groq",
                       help="Classification engine to use")
    parser.add_argument("--max-count", type=int, help="Maximum number of cross rulings to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Enable training data collection
    os.environ["COLLECT_TRAINING_DATA"] = "true"
    
    processor = CrossRulingsProcessor(engine_name=args.engine, debug=args.debug)
    
    try:
        summary = processor.process_cross_rulings_dataset(
            args.cross_rulings_file, 
            args.max_count
        )
        
        print(f"\n{'='*50}")
        print("PROCESSING SUMMARY")
        print(f"{'='*50}")
        print(f"Total processed: {summary['total_processed']}")
        print(f"Successful: {summary['successful']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        
        return 0 if summary['success_rate'] > 0 else 1
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
