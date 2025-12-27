"""
Cross Rulings Injector Module

Provides efficient lookup and injection of cross ruling examples into classification prompts.
Uses binary search for O(log n) time complexity on sorted HTS codes.
Uses keyword-based filtering for fast, reliable relevance scoring.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Tuple
from bisect import bisect_left, bisect_right
from rank_bm25 import BM25Okapi

class CrossRulingsInjector:
    """
    Manages cross rulings dataset and provides efficient lookup by HTS code.
    """
    
    def __init__(self, dataset_path: str = None):
        """
        Initialize the injector with cross rulings dataset.
        
        Args:
            dataset_path: Path to cross_rulings_dataset.json file
        """
        self.logger = logging.getLogger(__name__)
        self.cross_rulings = []
        self.sorted_codes = []
        self.bm25 = None  # BM25 index for text search
        self.tokenized_corpus = []  # Tokenized documents for BM25
        
        # Common stop words to ignore in keyword matching
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'or', 'not', 'no', 'other', 'than'
        }
        
        # Default path if not provided
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent / "cross_rulings.json"
        
        self.load_dataset(dataset_path)
    
    def load_dataset(self, dataset_path: str):
        """
        Load and index the cross rulings dataset.
        Sorts codes for efficient binary search.
        """
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.cross_rulings = json.load(f)
            
            # Sort by normalized HTS code for binary search
            self.cross_rulings.sort(key=lambda x: self.normalize_code(x.get('hts_code', '')))
            
            # Extract sorted normalized codes for binary search
            self.sorted_codes = [self.normalize_code(ruling.get('hts_code', '')) for ruling in self.cross_rulings]
            
            # Build BM25 index for text search
            self._build_bm25_index()
            
            self.logger.info(f"Loaded {len(self.cross_rulings)} cross rulings with BM25 index")
            
        except FileNotFoundError:
            self.logger.warning(f"Cross rulings dataset not found at {dataset_path}")
            self.cross_rulings = []
            self.sorted_codes = []
        except Exception as e:
            self.logger.error(f"Error loading cross rulings: {e}")
            self.cross_rulings = []
            self.sorted_codes = []
    
    def normalize_code(self, code: str) -> str:
        """
        Normalize HTS code by removing dots for comparison.
        
        Args:
            code: HTS code to normalize
            
        Returns:
            Normalized code without dots
        """
        return code.replace(".", "").strip()
    
    def find_exact_match(self, code: str) -> Optional[Dict]:
        """
        Find exact match for the given HTS code using binary search.
        
        Args:
            code: HTS code to search for
            
        Returns:
            Cross ruling dict if found, None otherwise
        """
        if not self.sorted_codes:
            return None
        
        normalized_target = self.normalize_code(code)
        
        # Binary search for exact match
        left = 0
        right = len(self.sorted_codes) - 1
        
        while left <= right:
            mid = (left + right) // 2
            # sorted_codes already contains normalized codes
            mid_code = self.sorted_codes[mid]
            
            if mid_code == normalized_target:
                return self.cross_rulings[mid]
            elif mid_code < normalized_target:
                left = mid + 1
            else:
                right = mid - 1
        
        return None
    
    def find_prefix_matches(self, code: str) -> List[Dict]:
        """
        Find all cross rulings that start with the given code prefix.
        Uses binary search to find the range efficiently.
        
        Args:
            code: HTS code prefix to search for
            
        Returns:
            List of matching cross rulings
        """
        if not self.sorted_codes:
            return []
        
        normalized_prefix = self.normalize_code(code)
        matches = []
        
        # Find the range of codes that start with this prefix
        # Using binary search to find the first and last matching indices
        
        # Find first index where code >= prefix
        left_idx = bisect_left(self.sorted_codes, normalized_prefix)
        
        # Collect all codes that start with the prefix
        for i in range(left_idx, len(self.sorted_codes)):
            # sorted_codes already contains normalized codes
            if self.sorted_codes[i].startswith(normalized_prefix):
                matches.append(self.cross_rulings[i])
            else:
                # Since codes are sorted, we can stop once we pass the prefix range
                break
        
        return matches
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """
        Extract meaningful keywords from text for matching.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            Set of lowercase keywords
        """
        if not text:
            return set()
        
        # Convert to lowercase and split by non-alphanumeric characters
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter out stop words and very short words
        keywords = {word for word in words if word not in self.stop_words and len(word) > 2}
        
        # Add compound terms (e.g., "desktop computer" -> also add "desktop_computer")
        text_lower = text.lower()
        compound_terms = [
            'desktop computer', 'laptop computer', 'tablet computer',
            'mobile phone', 'cell phone', 'smart phone', 'smartphone',
            'water filter', 'air filter', 'oil filter',
            'steam boiler', 'water boiler', 'gas boiler',
            'electric motor', 'combustion engine', 'diesel engine',
            'video game', 'gaming console', 'game controller'
        ]
        
        for term in compound_terms:
            if term in text_lower:
                keywords.add(term.replace(' ', '_'))
        
        return keywords
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 processing.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter out stop words and very short words
        tokens = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return tokens
    
    def _build_bm25_index(self):
        """
        Build BM25 index from the cross rulings dataset.
        """
        if not self.cross_rulings:
            return
        
        # Tokenize all ruling descriptions using ruling_info field
        self.tokenized_corpus = []
        for ruling in self.cross_rulings:
            description = ruling.get('ruling_info', ruling.get('full_description', ''))
            tokens = self._tokenize_text(description)
            self.tokenized_corpus.append(tokens)
        
        # Build BM25 index
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self.logger.info(f"Built BM25 index with {len(self.tokenized_corpus)} documents")
    
    def _calculate_keyword_score(self, product_keywords: Set[str], ruling_text: str) -> float:
        """
        Calculate relevance score based on keyword matching.
        
        Args:
            product_keywords: Keywords from the product description
            ruling_text: Text of the ruling to score
            
        Returns:
            Relevance score (higher is more relevant)
        """
        if not product_keywords or not ruling_text:
            return 0.0
        
        ruling_text_lower = ruling_text.lower()
        ruling_keywords = self._extract_keywords(ruling_text)
        
        score = 0.0
        
        # Exact keyword matches (weight: 2.0)
        exact_matches = product_keywords & ruling_keywords
        score += len(exact_matches) * 2.0
        
        # Substring matches in the ruling text (weight: 1.0)
        for keyword in product_keywords:
            if keyword not in exact_matches and keyword in ruling_text_lower:
                score += 1.0
        
        # Boost score for multiple word matches
        if len(exact_matches) >= 2:
            score *= 1.5  # 50% boost for multiple matches
        
        # Penalize if ruling contains many unrelated technical terms
        negative_indicators = {
            'boiler', 'furnace', 'turbine', 'pump', 'valve', 'pipe',
            'textile', 'fabric', 'yarn', 'thread', 'weaving',
            'food', 'beverage', 'edible', 'meat', 'vegetable',
            'chemical', 'pharmaceutical', 'medicine', 'drug'
        }
        
        # Only penalize if product keywords don't include these terms
        for neg_word in negative_indicators:
            if neg_word not in product_keywords and neg_word in ruling_text_lower:
                score *= 0.8  # 20% penalty for each unrelated domain
        
        return score
    
    def _filter_rulings_by_bm25(self, product_text: str, rulings: List[Dict]) -> List[Tuple[int, float]]:
        """
        Filter and score rulings using BM25 relevance scoring.
        
        Args:
            product_text: The product being classified
            rulings: List of cross rulings to evaluate
            
        Returns:
            List of (index, score) tuples for relevant rulings, sorted by score
        """
        if not product_text or not rulings or not self.bm25:
            return []
        
        # Tokenize the query
        query_tokens = self._tokenize_text(product_text)
        
        if not query_tokens:
            # If no tokens extracted, return first few rulings
            return [(i, 1.0) for i in range(min(3, len(rulings)))]
        
        # Create a mapping from ruling to its index in the full corpus
        ruling_to_corpus_idx = {}
        for i, ruling in enumerate(rulings):
            ruling_code = ruling.get('hts_code', '')
            # Find this ruling in the full corpus
            for j, corpus_ruling in enumerate(self.cross_rulings):
                if corpus_ruling.get('hts_code', '') == ruling_code:
                    ruling_to_corpus_idx[i] = j
                    break
        
        # Get BM25 scores for all documents in the corpus
        all_scores = self.bm25.get_scores(query_tokens)
        
        # Extract scores for our specific rulings
        scored_rulings = []
        for ruling_idx, corpus_idx in ruling_to_corpus_idx.items():
            if corpus_idx < len(all_scores):
                bm25_score = all_scores[corpus_idx]
                
                # Apply additional relevance filters  
                ruling_text = rulings[ruling_idx].get('ruling_info', rulings[ruling_idx].get('full_description', ''))
                
                # Boost score for exact phrase matches
                product_text_lower = product_text.lower()
                if any(phrase in ruling_text.lower() for phrase in [product_text_lower]):
                    bm25_score *= 1.5
                
                # Apply negative indicators penalty
                negative_indicators = {
                    'boiler', 'furnace', 'turbine', 'pump', 'valve', 'pipe',
                    'textile', 'fabric', 'yarn', 'thread', 'weaving',
                    'food', 'beverage', 'edible', 'meat', 'vegetable',
                    'chemical', 'pharmaceutical', 'medicine', 'drug'
                }
                
                product_keywords = self._extract_keywords(product_text)
                for neg_word in negative_indicators:
                    if neg_word not in product_keywords and neg_word in ruling_text.lower():
                        bm25_score *= 0.8  # 20% penalty for each unrelated domain
                
                # Only include rulings with positive scores
                if bm25_score > 0:
                    scored_rulings.append((ruling_idx, bm25_score))
        
        # Sort by score (highest first)
        scored_rulings.sort(key=lambda x: x[1], reverse=True)
        
        # Log filtering results
        self.logger.debug(f"BM25 filtering: {len(scored_rulings)} relevant out of {len(rulings)} rulings")
        
        return scored_rulings
    
    def _filter_rulings_by_keywords(self, product_text: str, rulings: List[Dict]) -> List[Tuple[int, float]]:
        """
        Filter and score rulings based on relevance.
        Now uses BM25 if available, falls back to keyword matching.
        
        Args:
            product_text: The product being classified
            rulings: List of cross rulings to evaluate
            
        Returns:
            List of (index, score) tuples for relevant rulings, sorted by score
        """
        if not product_text or not rulings:
            return []
        
        # Use BM25 if available, otherwise fall back to keyword matching
        if self.bm25:
            return self._filter_rulings_by_bm25(product_text, rulings)
        
        # Fallback to original keyword matching
        product_keywords = self._extract_keywords(product_text)
        
        if not product_keywords:
            # If no keywords extracted, return first few rulings
            return [(i, 1.0) for i in range(min(3, len(rulings)))]
        
        # Score each ruling
        scored_rulings = []
        for i, ruling in enumerate(rulings):
            ruling_text = ruling.get('full_description', '')
            score = self._calculate_keyword_score(product_keywords, ruling_text)
            
            # Only include rulings with positive scores
            if score > 0:
                scored_rulings.append((i, score))
        
        # Sort by score (highest first)
        scored_rulings.sort(key=lambda x: x[1], reverse=True)
        
        # Log filtering results
        self.logger.info(f"Keyword filtering: {len(scored_rulings)} relevant out of {len(rulings)} rulings")
        
        return scored_rulings
    
    def _get_max_prefix_candidates(self, code: str, total_matches: int) -> int:
        """
        Determine how many prefix candidates to consider based on code specificity.
        
        Args:
            code: HTS code being searched
            total_matches: Total number of prefix matches found
            
        Returns:
            Maximum number of candidates to consider for filtering
        """
        # Normalize code length (remove dots)
        normalized_code = self.normalize_code(code)
        code_length = len(normalized_code)
        
        if code_length <= 2:  # Chapter level (e.g., "84")
            # Very broad - limit to prevent irrelevant matches
            return min(15, total_matches)
        elif code_length <= 4:  # Heading level (e.g., "8471")
            # Moderately specific - allow more candidates
            return min(25, total_matches)
        elif code_length <= 6:  # Subheading level (e.g., "847130")
            # More specific - allow even more candidates
            return min(35, total_matches)
        else:  # Statistical suffix level (e.g., "8471300100")
            # Very specific - allow most candidates
            return min(50, total_matches)
    
    def _fallback_for_broad_codes(self, code: str, product_text: str, max_results: int) -> List[Dict]:
        """
        Fallback strategy for broad codes that don't match keywords.
        Try to find more specific subcodes that might be relevant.
        
        Args:
            code: Broad HTS code (usually 2-digit chapter)
            product_text: Product description
            max_results: Maximum results to return
            
        Returns:
            List of cross rulings from more specific subcodes
        """
        if len(self.normalize_code(code)) > 2:
            return []  # Only apply to chapter-level codes
        
        # Extract key terms from product text to guide subcode selection
        product_keywords = self._extract_keywords(product_text)
        
        # Define common subcode patterns for major chapters
        subcode_patterns = {
            '84': ['8471', '8473', '8528'],  # Computers, parts, monitors
            '85': ['8517', '8518', '8519'],  # Telecom, audio equipment
            '87': ['8703', '8704', '8708'],  # Vehicles and parts
            '90': ['9013', '9014', '9015'],  # Optical instruments
        }
        
        if code not in subcode_patterns:
            return []
        
        # Try each subcode pattern
        all_results = []
        for subcode in subcode_patterns[code]:
            subcode_results = self.lookup_cross_rulings(subcode, product_text, max_results)
            all_results.extend(subcode_results)
        
        # Remove duplicates and return top results
        seen_codes = set()
        unique_results = []
        for result in all_results:
            hts_code = result.get('hts_code', '')
            if hts_code not in seen_codes:
                seen_codes.add(hts_code)
                unique_results.append(result)
                if len(unique_results) >= max_results:
                    break
        
        return unique_results
    
    def lookup_cross_rulings(self, code: str, product_text: str = None, max_results: int = 3) -> List[Dict]:
        """
        Look up cross rulings for a given HTS code with optional keyword-based relevance filtering.
        
        Args:
            code: HTS code to look up
            product_text: Product description for relevance filtering (optional)
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant cross rulings (up to max_results)
        """
        # Get all potential rulings
        all_rulings = []
        
        # Try exact match first
        exact_match = self.find_exact_match(code)
        if exact_match:
            all_rulings.append({
                "hts_code": exact_match.get("hts_code", ""),
                "full_description": exact_match.get("full_description", "")
            })
        
        # Add prefix matches with smart limiting based on code specificity
        prefix_matches = self.find_prefix_matches(code)
        
        # Determine how many prefix matches to consider based on code length and specificity
        max_prefix_candidates = self._get_max_prefix_candidates(code, len(prefix_matches))
        
        for match in prefix_matches[:max_prefix_candidates]:
            match_code = match.get("hts_code", "")
            if not any(r["hts_code"] == match_code for r in all_rulings):
                all_rulings.append({
                    "hts_code": match_code,
                    "full_description": match.get("full_description", "")
                })
        
        # If no product text provided, return unfiltered results
        if not product_text:
            return all_rulings[:max_results]
        
        # Use keyword-based filtering
        if len(all_rulings) > 0:
            # Get scored rulings
            scored_rulings = self._filter_rulings_by_keywords(product_text, all_rulings)
            
            # Extract the top relevant rulings
            relevant_rulings = []
            for idx, score in scored_rulings:
                if idx < len(all_rulings):
                    relevant_rulings.append(all_rulings[idx])
                    if len(relevant_rulings) >= max_results:
                        break
            
            # If we have any relevant rulings, return them
            if relevant_rulings:
                return relevant_rulings
            
            # If all were filtered out but we have a very broad code, try fallback strategy
            if len(self.normalize_code(code)) <= 2 and not relevant_rulings:
                self.logger.info(f"No keyword matches for broad code '{code}', trying fallback strategy")
                return self._fallback_for_broad_codes(code, product_text, max_results)
            
            # If all were filtered out (no keyword matches), return empty list
            self.logger.debug(f"No keyword matches found for '{product_text}' in code {code}")
            return []
        
        return []
    
    def inject_cross_rulings_into_children(self, children: List[Dict], product_text: str = None) -> List[Dict]:
        """
        Inject cross ruling examples into a list of classification children.
        
        Args:
            children: List of child classification options
            product_text: Product description for relevance filtering (optional)
            
        Returns:
            Children list with cross_rulings field added to each child
        """
        if not children:
            return children
        
        enhanced_children = []
        
        for child in children:
            # Create a copy to avoid modifying original
            enhanced_child = child.copy()
            
            # Look up cross rulings for this child's code
            child_code = child.get("code", "")
            if child_code:
                # Pass product_text for relevance filtering
                cross_rulings = self.lookup_cross_rulings(child_code, product_text)
                if cross_rulings:
                    enhanced_child["cross_rulings"] = cross_rulings
                    self.logger.debug(f"Injected {len(cross_rulings)} cross rulings for code {child_code}")
            
            enhanced_children.append(enhanced_child)
        
        return enhanced_children
    
    def inject_into_classification_tree(self, classification_tree: Dict, product_text: str = None) -> Dict:
        """
        Inject cross rulings into a classification tree structure.
        
        Args:
            classification_tree: Classification tree with children
            product_text: Product description for relevance filtering (optional)
            
        Returns:
            Enhanced classification tree with cross rulings
        """
        if not classification_tree:
            return classification_tree
        
        # Create a copy to avoid modifying original
        enhanced_tree = classification_tree.copy()
        
        # Inject into children if present
        if "children" in enhanced_tree:
            enhanced_tree["children"] = self.inject_cross_rulings_into_children(
                enhanced_tree["children"],
                product_text
            )
        
        return enhanced_tree

# Global singleton instance
_injector_instance = None

def get_injector() -> CrossRulingsInjector:
    """
    Get or create the global cross rulings injector instance.
    
    Returns:
        CrossRulingsInjector singleton instance
    """
    global _injector_instance
    if _injector_instance is None:
        _injector_instance = CrossRulingsInjector()
    return _injector_instance

def inject_cross_rulings(data: Any) -> Any:
    """
    Convenience function to inject cross rulings into various data structures.
    
    Args:
        data: Data structure to enhance (dict with classification_tree, list of children, etc.)
        
    Returns:
        Enhanced data structure with cross rulings
    """
    injector = get_injector()
    
    if isinstance(data, dict):
        # Check if it's a classification tree
        if "children" in data:
            return injector.inject_into_classification_tree(data)
        # Check if it has a classification_tree field
        elif "classification_tree" in data:
            data = data.copy()
            data["classification_tree"] = injector.inject_into_classification_tree(
                data["classification_tree"]
            )
            return data
    elif isinstance(data, list):
        # Assume it's a list of children
        return injector.inject_cross_rulings_into_children(data)
    
    return data
