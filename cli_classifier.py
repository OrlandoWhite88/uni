#!/usr/bin/env python3
"""
HTS Classifier CLI Tool

A command-line interface for testing the HTS classification system locally
without needing to deploy to Vercel or use the web interface.

Usage:
    python cli_classifier.py                    # Interactive mode
    python cli_classifier.py --product "shoes"  # Single classification
    python cli_classifier.py --compare          # Compare single vs multi-hypothesis
    python cli_classifier.py --batch file.txt   # Batch classify from file

Features:
- Interactive classification with questions
- Single-shot classification
- Multi-hypothesis path exploration
- Comparison mode (single vs multi-hypothesis)
- Batch processing
- Session history saving
- Detailed logging and debugging
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the api directory to the Python path
script_dir = Path(__file__).parent
api_dir = script_dir / "api"
sys.path.insert(0, str(api_dir))

# Available engines mapping
AVAILABLE_ENGINES = {
    "cerebras": "cerebras_tree_engine",
    "groq": "groq_tree_engine", 
    "tree": "tree_engine",
    "gemini": "gemini_tree_engine"
}

def import_engine(engine_name: str):
    """Import the specified engine module and return the HTSTree class"""
    if engine_name not in AVAILABLE_ENGINES:
        available = ", ".join(AVAILABLE_ENGINES.keys())
        print(f"{Colors.FAIL}Error: Unknown engine '{engine_name}'. Available engines: {available}{Colors.ENDC}")
        sys.exit(1)
    
    module_name = AVAILABLE_ENGINES[engine_name]
    
    try:
        # Import as part of the api package to support relative imports
        module = __import__(f"api.{module_name}", fromlist=[module_name])
        if not hasattr(module, 'HTSTree'):
            print(f"{Colors.FAIL}Error: Engine '{engine_name}' does not have an HTSTree class{Colors.ENDC}")
            sys.exit(1)
        
        print(f"{Colors.OKGREEN}✓ Using {engine_name} engine ({module_name}){Colors.ENDC}")
        return module.HTSTree
    except ImportError as e:
        print(f"{Colors.FAIL}Error importing {engine_name} engine: {e}{Colors.ENDC}")
        print("Make sure you're running this script from the project root directory.")
        print("Also ensure all required API keys are set (GROQ_API_KEY for Groq engine, etc.)")
        sys.exit(1)

# Default engine will be set after parsing arguments
HTSTree = None

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class HTSClassifierCLI:
    def __init__(self, debug: bool = False, quiet: bool = False, save_trajectory: bool = False):
        self.debug = debug
        self.quiet = quiet
        self.save_trajectory = save_trajectory
        self.setup_logging()
        self.hts_tree = None
        self.session_history = []
        self.load_hts_data()
        
    def setup_logging(self):
        """Configure logging based on debug mode"""
        if self.quiet:
            # In quiet mode, only show errors
            level = logging.ERROR
        elif self.debug:
            level = logging.DEBUG
        else:
            level = logging.INFO
            
        # Configure file logging (always detailed)
        file_handler = logging.FileHandler('classifier_cli.log')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Configure console logging (respects quiet/debug modes)
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
            
        # In quiet mode, suppress almost everything except errors
        if self.quiet:
            logging.getLogger('google_genai').setLevel(logging.ERROR)
            logging.getLogger('root').setLevel(logging.ERROR)
            
        self.logger = logging.getLogger(__name__)
        
    def load_hts_data(self):
        """Load HTS data and build the tree"""
        try:
            hts_data_file = script_dir / "api" / "hts_data.json"
            if not hts_data_file.exists():
                raise FileNotFoundError(f"HTS data file not found: {hts_data_file}")
                
            if not self.quiet:
                print(f"{Colors.OKBLUE}Loading HTS data...{Colors.ENDC}")
            with open(hts_data_file, "r", encoding="utf-8") as f:
                hts_data = json.load(f)
                
            self.hts_tree = HTSTree()
            self.hts_tree.build_from_json(hts_data)
            
            if not self.quiet:
                print(f"{Colors.OKGREEN}✓ Loaded HTS tree with {len(self.hts_tree.code_index)} HTS codes and {len(self.hts_tree.node_index)} total nodes.{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.FAIL}Error loading HTS data: {e}{Colors.ENDC}")
            sys.exit(1)
            
    def print_header(self):
        """Print CLI header"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
        print("HTS CLASSIFIER CLI TOOL")
        print(f"{'='*60}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Test your HTS classifications locally without deployment{Colors.ENDC}\n")
        
    def print_classification_result(self, result: Dict[str, Any], title: str = "Classification Result"):
        """Pretty print classification results"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{title}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*len(title)}{Colors.ENDC}")
        
        # Check if this is a clarification question (tree engine returns "final": False with "clarification_question")
        if not result.get("final", True) and result.get("clarification_question"):
            question = result.get("clarification_question")
            print(f"{Colors.WARNING}❓ Clarification needed:{Colors.ENDC}")
            
            # Handle both ClarificationQuestion object and dict formats
            if hasattr(question, 'question_text'):
                # ClarificationQuestion object
                print(f"   {question.question_text}")
                options = question.options
            else:
                # Dictionary format
                print(f"   {question.get('question_text', 'No question text')}")
                options = question.get("options", [])
            
            if options:
                print(f"\n{Colors.OKBLUE}Options:{Colors.ENDC}")
                for i, option in enumerate(options, 1):
                    if isinstance(option, dict):
                        print(f"   {i}. {option.get('text', option)}")
                    else:
                        print(f"   {i}. {option}")
        else:
            # Final classification - handle both old and new formats
            classification = result.get("classification", {})
            if classification:
                # New format: result["classification"]["code"]
                final_code = classification.get("code")
                full_path = classification.get("path")
                confidence = classification.get("confidence", 0)
            else:
                # Old format: result["final_code"]
                final_code = result.get("final_code")
                full_path = result.get("full_path")
                confidence = result.get("confidence", 0)
            
            print(f"{Colors.OKGREEN}✓ Final Classification:{Colors.ENDC}")
            print(f"   Code: {Colors.BOLD}{final_code}{Colors.ENDC}")
            print(f"   Path: {full_path}")
            print(f"   Confidence: {confidence:.3f}")
            
            # Multi-hypothesis info
            multi_info = result.get("multi_hypothesis_info", {})
            if multi_info:
                print(f"\n{Colors.OKCYAN}Multi-Hypothesis Analysis:{Colors.ENDC}")
                print(f"   Paths explored: {multi_info.get('total_paths', 0)}")
                print(f"   Active paths: {multi_info.get('active_paths', 0)}")
                print(f"   Completed paths: {multi_info.get('completed_paths', 0)}")
                
                # Show top paths
                top_paths = multi_info.get("top_paths", [])
                if top_paths:
                    print(f"\n{Colors.OKBLUE}Top Classification Paths:{Colors.ENDC}")
                    for i, path in enumerate(top_paths[:3], 1):
                        print(f"   {i}. {path.get('final_code')} (conf: {path.get('cumulative_confidence', 0):.3f})")
                        print(f"      {path.get('full_path', '')}")
            
            # Show conversation history if available
            conversation = result.get("conversation", [])
            if conversation:
                print(f"\n{Colors.OKCYAN}Conversation History:{Colors.ENDC}")
                for entry in conversation:
                    print(f"   Q: {entry.get('question', '')}")
                    print(f"   A: {entry.get('answer', '')}")
                    
    def interactive_classify(self, product: str, use_multi_hypothesis: bool = True, hypothesis_count: int = 3):
        """Run interactive classification with questions"""
        if not self.quiet:
            print(f"\n{Colors.OKBLUE}Starting interactive classification for: {Colors.BOLD}{product}{Colors.ENDC}")
            print(f"Multi-hypothesis mode: {Colors.BOLD}{'ON' if use_multi_hypothesis else 'OFF'}{Colors.ENDC}")
            if use_multi_hypothesis:
                print(f"{Colors.OKCYAN}🔍 This will explore {hypothesis_count} hypothesis paths and ask clarifying questions...{Colors.ENDC}")
            else:
                print(f"{Colors.OKCYAN}🔍 This will use single-path classification with clarifying questions...{Colors.ENDC}")
        
        try:
            # Start classification
            result = self.hts_tree.start_classification(
                product=product,
                interactive=True,
                max_questions=5,  # Allow more questions in CLI
                use_multi_hypothesis=use_multi_hypothesis,
                hypothesis_count=hypothesis_count
            )
            
            session = {
                "product": product,
                "start_time": datetime.now().isoformat(),
                "interactions": [],
                "final_result": None
            }
            
            # Handle interactive loop - check for clarification questions
            while not result.get("final", True) and result.get("clarification_question"):
                self.print_classification_result(result, "Clarification Needed")
                
                question = result.get("clarification_question")
                
                # Handle both ClarificationQuestion object and dict formats
                if hasattr(question, 'question_text'):
                    # ClarificationQuestion object
                    options = question.options
                    question_text = question.question_text
                else:
                    # Dictionary format
                    options = question.get("options", [])
                    question_text = question.get("question_text", "")
                
                # Get user input
                if options:
                    while True:
                        try:
                            choice = input(f"\n{Colors.OKCYAN}Enter your choice (1-{len(options)}) or type your answer: {Colors.ENDC}")
                            
                            # Check if it's a number (option selection)
                            if choice.isdigit():
                                choice_num = int(choice)
                                if 1 <= choice_num <= len(options):
                                    option = options[choice_num - 1]
                                    if isinstance(option, dict):
                                        answer = option.get('text', option)
                                    else:
                                        answer = str(option)
                                    break
                                else:
                                    print(f"{Colors.WARNING}Please enter a number between 1 and {len(options)}{Colors.ENDC}")
                            else:
                                # Free text answer
                                answer = choice
                                break
                        except KeyboardInterrupt:
                            print(f"\n{Colors.WARNING}Classification cancelled by user{Colors.ENDC}")
                            return None
                else:
                    # Free text question
                    try:
                        answer = input(f"\n{Colors.OKCYAN}Your answer: {Colors.ENDC}")
                    except KeyboardInterrupt:
                        print(f"\n{Colors.WARNING}Classification cancelled by user{Colors.ENDC}")
                        return None
                
                # Record interaction
                session["interactions"].append({
                    "question": question_text,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Continue classification
                result = self.hts_tree.continue_classification(
                    state=result.get("state", {}),
                    answer=answer,
                    interactive=True,
                    max_questions=5
                )
            
            # Final result
            session["final_result"] = result
            session["end_time"] = datetime.now().isoformat()
            self.session_history.append(session)
            
            self.print_classification_result(result, "Final Classification")
            
            # Save trajectory if enabled
            self.extract_and_save_trajectory(result, product)
            
            return result
            
        except Exception as e:
            print(f"{Colors.FAIL}Error during classification: {e}{Colors.ENDC}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
            
    def single_classify(self, product: str, use_multi_hypothesis: bool = True, hypothesis_count: int = 3):
        """Run single-shot classification without questions"""
        if not self.quiet:
            print(f"\n{Colors.OKBLUE}Running single-shot classification for: {Colors.BOLD}{product}{Colors.ENDC}")
            if use_multi_hypothesis:
                print(f"{Colors.OKCYAN}🔍 Exploring {hypothesis_count} hypothesis paths...{Colors.ENDC}")
            else:
                print(f"{Colors.OKCYAN}🔍 Using single-path classification...{Colors.ENDC}")
        
        try:
            result = self.hts_tree.start_classification(
                product=product,
                interactive=False,
                max_questions=0,
                use_multi_hypothesis=use_multi_hypothesis,
                hypothesis_count=hypothesis_count
            )
            
            if not self.quiet:
                self.print_classification_result(result)
            
            # Save trajectory if enabled
            self.extract_and_save_trajectory(result, product)
            
            return result
            
        except Exception as e:
            print(f"{Colors.FAIL}Error during classification: {e}{Colors.ENDC}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
            
    def compare_modes(self, product: str, hypothesis_count: int = 3):
        """Compare single-path vs multi-hypothesis classification"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}COMPARISON MODE{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*15}{Colors.ENDC}")
        print(f"Product: {Colors.BOLD}{product}{Colors.ENDC}\n")
        
        # Single-path classification
        print(f"{Colors.OKBLUE}Running single-path classification...{Colors.ENDC}")
        single_result = self.single_classify(product, use_multi_hypothesis=False)
        
        print(f"\n{Colors.OKBLUE}Running multi-hypothesis classification...{Colors.ENDC}")
        multi_result = self.single_classify(product, use_multi_hypothesis=True, hypothesis_count=hypothesis_count)
        
        # Comparison summary
        print(f"\n{Colors.HEADER}{Colors.BOLD}COMPARISON SUMMARY{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*18}{Colors.ENDC}")
        
        if single_result and multi_result:
            single_code = single_result.get("final_code", "N/A")
            multi_code = multi_result.get("final_code", "N/A")
            
            print(f"Single-path result:     {Colors.BOLD}{single_code}{Colors.ENDC}")
            print(f"Multi-hypothesis result: {Colors.BOLD}{multi_code}{Colors.ENDC}")
            
            if single_code == multi_code:
                print(f"{Colors.OKGREEN}✓ Both methods agree!{Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}⚠ Different results - multi-hypothesis may be more accurate{Colors.ENDC}")
                
            # Show confidence comparison
            single_conf = single_result.get("confidence", 0)
            multi_conf = multi_result.get("confidence", 0)
            print(f"\nConfidence comparison:")
            print(f"Single-path:     {single_conf:.3f}")
            print(f"Multi-hypothesis: {multi_conf:.3f}")
            
        return single_result, multi_result
        
    def batch_classify(self, file_path: str, use_multi_hypothesis: bool = True):
        """Classify multiple products from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                products = [line.strip() for line in f if line.strip()]
                
            print(f"\n{Colors.OKBLUE}Batch classifying {len(products)} products...{Colors.ENDC}")
            
            results = []
            for i, product in enumerate(products, 1):
                print(f"\n{Colors.OKCYAN}[{i}/{len(products)}] {product}{Colors.ENDC}")
                result = self.single_classify(product, use_multi_hypothesis=use_multi_hypothesis)
                
                if result:
                    results.append({
                        "product": product,
                        "final_code": result.get("final_code"),
                        "confidence": result.get("confidence", 0),
                        "full_path": result.get("full_path")
                    })
                else:
                    results.append({
                        "product": product,
                        "error": "Classification failed"
                    })
                    
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"batch_results_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            print(f"\n{Colors.OKGREEN}✓ Batch classification complete. Results saved to: {output_file}{Colors.ENDC}")
            return results
            
        except Exception as e:
            print(f"{Colors.FAIL}Error in batch classification: {e}{Colors.ENDC}")
            return None
            
    def save_session_history(self):
        """Save session history to file"""
        if not self.session_history:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_history_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.session_history, f, indent=2, ensure_ascii=False)
            print(f"{Colors.OKGREEN}Session history saved to: {filename}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.WARNING}Failed to save session history: {e}{Colors.ENDC}")
    
    def extract_and_save_trajectory(self, result: Dict[str, Any], product: str, gold_code: Optional[str] = None) -> Optional[List[str]]:
        """
        Extract the top 3 paths' trajectories from a classification result and save to separate files.
        
        Args:
            result: Classification result containing beam and state
            product: Product description
            gold_code: Optional gold/expected code for computing is_correct (for TreeRL training)
        
        Returns list of filenames if saved, None otherwise.
        """
        if not self.save_trajectory:
            return None
            
        state = result.get("state", {})
        beam = state.get("beam", [])
        
        if not beam or not isinstance(beam, list):
            print(f"{Colors.WARNING}No beam found in result (trajectory mode may be disabled){Colors.ENDC}")
            return None
        
        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_product = "".join(c if c.isalnum() or c in " -_" else "_" for c in product[:30])
        
        # Save top 3 trajectories (or however many are available)
        for rank, path_data in enumerate(beam[:3], 1):
            # Extract trajectory from path
            if isinstance(path_data, dict):
                trajectory = path_data.get("trajectory", [])
                path_id = path_data.get("path_id", f"path_{rank}")
                final_code = path_data.get("selection", {}).get("tariff") or path_data.get("selection", {}).get("subheading") or path_data.get("selection", {}).get("heading")
                cumulative_conf = path_data.get("cumulative_confidence", 0)
                classification_path = path_data.get("classification_path", [])
                full_path = " > ".join([f"{step.get('code', '')} - {step.get('description', '')}" for step in classification_path])
            elif hasattr(path_data, "trajectory"):
                trajectory = path_data.trajectory
                path_id = path_data.path_id
                final_code = path_data.selection.get("tariff") or path_data.selection.get("subheading") or path_data.selection.get("heading")
                cumulative_conf = path_data.cumulative_confidence
                full_path = " > ".join([f"{step.get('code', '')} - {step.get('description', '')}" for step in path_data.classification_path])
            else:
                continue
            
            if not trajectory:
                print(f"{Colors.WARNING}No trajectory found for path {rank}{Colors.ENDC}")
                continue
            
            # Extract path_trace for TreeRL training
            # This is a list of selected codes at each level for building the RL tree structure
            if isinstance(path_data, dict):
                classification_path_data = path_data.get("classification_path", [])
            elif hasattr(path_data, "classification_path"):
                classification_path_data = path_data.classification_path
            else:
                classification_path_data = []
            
            path_trace = [step.get("code", "") for step in classification_path_data if step.get("code")]
            
            # Compute is_correct for TreeRL training (only when gold_code is provided)
            is_correct = None
            if gold_code is not None:
                # Normalize codes for comparison (handle both string and int, strip dots)
                final_code_normalized = str(final_code).replace(".", "") if final_code else ""
                gold_code_normalized = str(gold_code).replace(".", "")
                is_correct = final_code_normalized == gold_code_normalized
            
            # Format the trajectory for human readability
            formatted_trajectory = {
                "product": product,
                "timestamp": datetime.now().isoformat(),
                "rank": rank,
                "path_id": path_id,
                "classification_result": {
                    "final_code": final_code,
                    "confidence": cumulative_conf,
                    "full_path": full_path
                },
                # TreeRL metadata - enables tree construction and advantage computation
                # Not needed for inference, only for RL training
                "path_trace": path_trace,
                "is_correct": is_correct,  # None if gold_code not provided, True/False otherwise
                "gold_code": gold_code,    # Reference for training
                "message_count": len(trajectory),
                "messages": trajectory
            }
            
            # Create filename with rank
            filename = f"trajectory_{safe_product}_rank{rank}_{timestamp}.json"
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(formatted_trajectory, f, indent=2, ensure_ascii=False)
                print(f"{Colors.OKGREEN}✓ Trajectory #{rank} saved to: {filename} ({len(trajectory)} messages){Colors.ENDC}")
                saved_files.append(filename)
            except Exception as e:
                print(f"{Colors.WARNING}Failed to save trajectory #{rank}: {e}{Colors.ENDC}")
        
        return saved_files if saved_files else None
            
    def interactive_menu(self):
        """Main interactive menu"""
        self.print_header()
        
        while True:
            print(f"\n{Colors.OKBLUE}{Colors.BOLD}MAIN MENU{Colors.ENDC}")
            print(f"{Colors.OKBLUE}{'='*9}{Colors.ENDC}")
            print("1. Interactive Classification (with questions)")
            print("2. Single-shot Classification (no questions)")
            print("3. Compare Single vs Multi-hypothesis")
            print("4. Batch Classification from file")
            print("5. View Session History")
            print("6. Save Session History")
            print("7. Toggle Debug Mode")
            print("0. Exit")
            
            try:
                choice = input(f"\n{Colors.OKCYAN}Enter your choice (0-7): {Colors.ENDC}")
                
                if choice == "0":
                    print(f"{Colors.OKGREEN}Goodbye!{Colors.ENDC}")
                    break
                elif choice == "1":
                    product = input(f"{Colors.OKCYAN}Enter product description: {Colors.ENDC}")
                    if product.strip():
                        self.interactive_classify(product.strip())
                elif choice == "2":
                    product = input(f"{Colors.OKCYAN}Enter product description: {Colors.ENDC}")
                    if product.strip():
                        self.single_classify(product.strip())
                elif choice == "3":
                    product = input(f"{Colors.OKCYAN}Enter product description: {Colors.ENDC}")
                    if product.strip():
                        self.compare_modes(product.strip())
                elif choice == "4":
                    file_path = input(f"{Colors.OKCYAN}Enter file path: {Colors.ENDC}")
                    if file_path.strip() and os.path.exists(file_path.strip()):
                        self.batch_classify(file_path.strip())
                    else:
                        print(f"{Colors.WARNING}File not found: {file_path}{Colors.ENDC}")
                elif choice == "5":
                    self.show_session_history()
                elif choice == "6":
                    self.save_session_history()
                elif choice == "7":
                    self.debug = not self.debug
                    self.setup_logging()
                    print(f"{Colors.OKGREEN}Debug mode: {'ON' if self.debug else 'OFF'}{Colors.ENDC}")
                else:
                    print(f"{Colors.WARNING}Invalid choice. Please try again.{Colors.ENDC}")
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.OKGREEN}Goodbye!{Colors.ENDC}")
                break
            except Exception as e:
                print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
                
    def show_session_history(self):
        """Display session history"""
        if not self.session_history:
            print(f"{Colors.WARNING}No session history available{Colors.ENDC}")
            return
            
        print(f"\n{Colors.HEADER}{Colors.BOLD}SESSION HISTORY{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*15}{Colors.ENDC}")
        
        for i, session in enumerate(self.session_history, 1):
            print(f"\n{Colors.OKBLUE}Session {i}:{Colors.ENDC}")
            print(f"  Product: {session.get('product', 'N/A')}")
            print(f"  Start: {session.get('start_time', 'N/A')}")
            print(f"  Interactions: {len(session.get('interactions', []))}")
            
            final_result = session.get('final_result', {})
            if final_result:
                print(f"  Final Code: {final_result.get('final_code', 'N/A')}")
                print(f"  Confidence: {final_result.get('confidence', 0):.3f}")

def main():
    parser = argparse.ArgumentParser(
        description="HTS Classifier CLI Tool - Test classifications locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_classifier.py                           # Interactive mode
  python cli_classifier.py --product "leather shoes" # Single classification
  python cli_classifier.py --product "shoes" --interactive  # Interactive classification
  python cli_classifier.py --compare --product "shoes"      # Compare modes
  python cli_classifier.py --batch products.txt             # Batch classify
  python cli_classifier.py --debug                          # Enable debug logging
  python cli_classifier.py --quiet --product "shoes"       # Quiet mode - minimal output
  python cli_classifier.py -t --product "fish"             # Save chat trajectory to file
        """
    )
    
    parser.add_argument("--product", "-p", help="Product to classify")
    parser.add_argument("--interactive", "-i", action="store_true", help="Use interactive mode with questions")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare single vs multi-hypothesis")
    parser.add_argument("--batch", "-b", help="Batch classify from file")
    parser.add_argument("--single-path", action="store_true", help="Use single-path mode (disable multi-hypothesis)")
    parser.add_argument("--hypothesis-count", type=int, default=3, help="Number of hypothesis paths (default: 3)")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode - minimal output")
    parser.add_argument("--save-history", action="store_true", help="Save session history on exit")
    parser.add_argument("--save-trajectory", "-t", action="store_true", 
                        help="Save the winning path's chat trajectory to a JSON file")
    parser.add_argument("--engine", "-e", choices=list(AVAILABLE_ENGINES.keys()), default="groq", 
                        help="Choose classification engine (default: groq)")
    
    args = parser.parse_args()
    
    # Import the selected engine and set it globally
    global HTSTree
    HTSTree = import_engine(args.engine)
    
    # Initialize CLI
    cli = HTSClassifierCLI(debug=args.debug, quiet=args.quiet, save_trajectory=args.save_trajectory)
    
    try:
        if args.batch:
            # Batch mode
            cli.batch_classify(args.batch, use_multi_hypothesis=not args.single_path)
        elif args.compare and args.product:
            # Compare mode
            cli.compare_modes(args.product, args.hypothesis_count)
        elif args.product:
            # Single product classification
            if args.interactive:
                cli.interactive_classify(args.product, use_multi_hypothesis=not args.single_path, hypothesis_count=args.hypothesis_count)
            else:
                cli.single_classify(args.product, use_multi_hypothesis=not args.single_path, hypothesis_count=args.hypothesis_count)
        else:
            # Interactive menu mode
            cli.interactive_menu()
            
    except KeyboardInterrupt:
        print(f"\n{Colors.OKGREEN}Operation cancelled by user{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}Unexpected error: {e}{Colors.ENDC}")
        if args.debug:
            import traceback
            traceback.print_exc()
    finally:
        if args.save_history:
            cli.save_session_history()

if __name__ == "__main__":
    main()
