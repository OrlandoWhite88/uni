"""
Tariff Lookup Engine
====================
Combines HTS JSON data (for current rates) with Excel data (for additional metadata).
JSON is the source of truth for rates, Excel provides supplementary information.
"""

import json
import logging
import os
import re
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class TariffLookupEngine:
    """
    Handles tariff lookups from both JSON and Excel sources.
    JSON provides current rates (source of truth).
    Excel provides additional metadata and country-specific details.
    """
    
    def __init__(self):
        self._excel_data = None
        self._excel_last_modified = 0
        self._excel_path = None
        self.tree = None
        
    def set_tree(self, tree):
        """Set the HTS tree instance for JSON lookups."""
        self.tree = tree
        logger.info(f"Tree set with {len(tree.code_index)} codes")
        
    def _load_excel_data(self):
        """Load or reload Excel data if needed."""
        # Find Excel file
        if self._excel_path is None:
            cwd = os.getcwd()
            possible_paths = [
                os.path.join(cwd, 'tarrif.xlsx'),
                'tarrif.xlsx',
                os.path.join(cwd, 'api', 'tarrif.xlsx'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tarrif.xlsx')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self._excel_path = path
                    logger.info(f"Found Excel file at: {self._excel_path}")
                    break
                    
            if not self._excel_path:
                logger.warning("Tarrif Excel file not found")
                return False
                
        # Check if reload needed
        if os.path.exists(self._excel_path):
            current_mtime = os.path.getmtime(self._excel_path)
            
            if self._excel_data is None or current_mtime > self._excel_last_modified:
                logger.info(f"Loading Excel file from: {self._excel_path}")
                
                # Load with pandas, preserving leading zeros
                df = pd.read_excel(self._excel_path, dtype={'hts8': str})
                
                # Convert to dictionary for fast lookups
                self._excel_data = {}
                for _, row in df.iterrows():
                    hts8 = str(row.get('hts8', '')).strip()
                    if hts8:
                        # Convert row to dict, handling NaN values
                        row_dict = {}
                        for key, value in row.items():
                            if pd.notna(value) and value != '' and str(value).lower() != 'nan':
                                if hasattr(value, 'strftime'):  # Date handling
                                    row_dict[key] = value.strftime('%Y-%m-%d')
                                else:
                                    row_dict[key] = value
                        self._excel_data[hts8] = row_dict
                        
                self._excel_last_modified = current_mtime
                logger.info(f"Excel loaded with {len(self._excel_data)} entries")
                return True
                
        return False
        
    def _normalize_hts_code(self, hts_code: str) -> str:
        """Normalize HTS code by removing dots and limiting length."""
        clean_code = hts_code.replace('.', '').replace(' ', '')
        # HTS codes can be up to 10 digits
        if len(clean_code) > 10:
            clean_code = clean_code[:10]
        return clean_code
        
    def _format_with_dots(self, code: str) -> str:
        """Format HTS code with standard dots (e.g., 0101.21.00.10)."""
        if len(code) >= 4:
            formatted = code[:4]
            if len(code) >= 6:
                formatted += '.' + code[4:6]
            if len(code) >= 8:
                formatted += '.' + code[6:8]
            if len(code) >= 10:
                formatted += '.' + code[8:10]
            return formatted
        return code
        
    def _get_code_variants(self, code: str) -> List[str]:
        """Get different formatting variants of an HTS code."""
        variants = [code]
        
        # Add dotted format
        if len(code) >= 4:
            variants.append(self._format_with_dots(code))
            
        # Add zero-padded variants for shorter codes
        if len(code) == 6:
            variants.append(code + "00")  # Try as 8-digit
            variants.append(self._format_with_dots(code + "00"))
        elif len(code) == 8:
            variants.append(code + "00")  # Try as 10-digit
            variants.append(self._format_with_dots(code + "00"))
            
        return variants
        
    def _get_json_data(self, hts_code: str) -> Optional[Dict[str, Any]]:
        """Get tariff data from JSON with inheritance."""
        if not self.tree:
            logger.error("Tree not initialized")
            return None
            
        clean_code = self._normalize_hts_code(hts_code)
        
        # Try exact match first
        node = None
        
        # Try all variants of the code
        for variant in self._get_code_variants(clean_code):
            if variant in self.tree.code_index:
                node = self.tree.code_index[variant]
                logger.info(f"Found exact match in JSON: {variant}")
                break
                
        # If not found, try progressive truncation
        if not node:
            search_code = clean_code
            while len(search_code) >= 2 and not node:
                for variant in self._get_code_variants(search_code):
                    if variant in self.tree.code_index:
                        node = self.tree.code_index[variant]
                        logger.info(f"Found parent match in JSON: {variant} for {hts_code}")
                        break
                # Truncate by 2 digits for next iteration
                if not node:
                    if len(search_code) == 10:
                        search_code = search_code[:8]
                    elif len(search_code) == 8:
                        search_code = search_code[:6]
                    elif len(search_code) == 6:
                        search_code = search_code[:4]
                    elif len(search_code) == 4:
                        search_code = search_code[:2]
                    else:
                        search_code = search_code[:-2]
                        
        if node:
            # Get inherited rates using existing methods
            return {
                "hts_code": node.htsno or clean_code,
                "description": self._clean_html(node.description),
                "general": node.get_inherited_general_rate(),
                "special": node.get_inherited_special_rate(),
                "other": node.get_inherited_other_rate(),
                "footnotes": node.get_inherited_footnotes(),
                "units": node.units,
                "superior": node.is_superior,
                "indent": node.indent,
                "node_type": node.get_node_type(),
                "matched_code": node.htsno,
                "match_type": "exact" if node.htsno == clean_code or node.htsno == self._format_with_dots(clean_code) else "parent"
            }
            
        return None
        
    def _get_excel_data(self, hts_code: str) -> Optional[Dict[str, Any]]:
        """Get tariff data from Excel."""
        # Ensure Excel data is loaded
        if self._excel_data is None:
            if not self._load_excel_data():
                return None
                
        clean_code = self._normalize_hts_code(hts_code)
        
        # Excel uses 8-digit codes, so truncate if longer
        search_code = clean_code[:8] if len(clean_code) > 8 else clean_code
        
        # Try progressively shorter codes
        while len(search_code) >= 2:
            if search_code in self._excel_data:
                logger.info(f"Found match in Excel: {search_code} for {hts_code}")
                result = self._excel_data[search_code].copy()
                result["excel_matched_code"] = search_code
                result["excel_match_type"] = "exact" if search_code == clean_code[:8] else "parent"
                return result
            # Truncate by 2 digits
            search_code = search_code[:-2]
            
        return None
        
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return text
        return re.sub(r'</?i>', '', text)
        
    def get_combined_tariff_info(self, hts_code: str) -> Dict[str, Any]:
        """
        Get combined tariff information from both JSON and Excel.
        
        Returns a dictionary with ALL Excel fields, but with JSON values
        overriding Excel values for rate fields (source of truth).
        """
        logger.info(f"Getting combined tariff info for: {hts_code}")
        
        # Get data from both sources
        json_data = self._get_json_data(hts_code)
        excel_data = self._get_excel_data(hts_code)
        
        # Handle case where neither source has data
        if not json_data and not excel_data:
            logger.warning(f"No data found for {hts_code} in either source")
            return {
                "error": "Not Found",
                "status_code": 404,
                "requested_code": hts_code,
                "message": "HTS code not found in either JSON or Excel data"
            }
            
        # Start with all Excel fields (if available) as the base
        if excel_data:
            result = excel_data.copy()
        else:
            # Initialize with empty values for all expected fields
            result = self._initialize_empty_result()
            
        # Override with JSON data (source of truth for rates)
        if json_data:
            # Map JSON fields to Excel field names
            json_to_excel_mapping = {
                "general": "mfn_text_rate",
                "special": "col1_special_text",
                "other": "col2_text_rate",
                "description": "brief_description",
                "units": "quantity_1_code"  # Note: units is an array in JSON
            }
            
            # Override Excel values with JSON values
            for json_field, excel_field in json_to_excel_mapping.items():
                if json_field in json_data:
                    value = json_data[json_field]
                    if json_field == "units" and isinstance(value, list):
                        # Handle units array
                        if len(value) > 0:
                            result["quantity_1_code"] = value[0] if value[0] else result.get("quantity_1_code")
                        if len(value) > 1:
                            result["quantity_2_code"] = value[1] if value[1] else result.get("quantity_2_code")
                    else:
                        result[excel_field] = value
                        
            # Add footnotes (may not have Excel equivalent)
            if json_data.get("footnotes"):
                footnotes_text = "; ".join([str(f) for f in json_data["footnotes"]])
                result["footnote_comment"] = footnotes_text
                
            # Store JSON-specific metadata
            result["_json_metadata"] = {
                "matched_code": json_data.get("matched_code"),
                "match_type": json_data.get("match_type"),
                "node_type": json_data.get("node_type"),
                "superior": json_data.get("superior"),
                "indent": json_data.get("indent")
            }
            
        # Ensure hts8 field is set correctly
        clean_code = self._normalize_hts_code(hts_code)
        result["hts8"] = clean_code[:8] if len(clean_code) >= 8 else clean_code
        
        # Add metadata about data sources
        result["_data_sources"] = {
            "requested_code": hts_code,
            "normalized_code": clean_code,
            "json_available": json_data is not None,
            "excel_available": excel_data is not None,
            "rates_from": "json" if json_data else "excel",
            "metadata_from": "excel" if excel_data else "json"
        }
        
        if json_data and excel_data:
            # Add comparison for verification (useful for debugging)
            result["_rate_comparison"] = {
                "general": {
                    "json": json_data.get("general"),
                    "excel": excel_data.get("mfn_text_rate")
                },
                "special": {
                    "json": json_data.get("special"),
                    "excel": excel_data.get("col1_special_text")
                },
                "other": {
                    "json": json_data.get("other"),
                    "excel": excel_data.get("col2_text_rate")
                }
            }
            
        return result
        
    def _initialize_empty_result(self) -> Dict[str, Any]:
        """Initialize result with all Excel fields as empty/null."""
        # All fields from the Excel file
        fields = [
            "hts8", "brief_description", "quantity_1_code", "quantity_2_code",
            "wto_binding_code", "mfn_text_rate", "mfn_rate_type_code", "mfn_ave",
            "mfn_ad_val_rate", "mfn_specific_rate", "mfn_other_rate",
            "col1_special_text", "col1_special_mod", "gsp_indicator", "gsp_ctry_excluded",
            "apta_indicator", "civil_air_indicator", "nafta_canada_ind", "nafta_mexico_ind",
            "mexico_rate_type_code", "mexico_ad_val_rate", "mexico_specific_rate",
            "cbi_indicator", "cbi_ad_val_rate", "cbi_specific_rate", "agoa_indicator",
            "cbtpa_indicator", "cbtpa_rate_type_code", "cbtpa_ad_val_rate", "cbtpa_specific_rate",
            "israel_fta_indicator", "atpa_indicator", "atpa_ad_val_rate", "atpa_specific_rate",
            "atpdea_indicator", "jordan_indicator", "jordan_rate_type_code",
            "jordan_ad_val_rate", "jordan_specific_rate", "jordan_other_rate",
            "singapore_indicator", "singapore_rate_type_code", "singapore_ad_val_rate",
            "singapore_specific_rate", "singapore_other_rate", "chile_indicator",
            "chile_rate_type_code", "chile_ad_val_rate", "chile_specific_rate", "chile_other_rate",
            "morocco_indicator", "morocco_rate_type_code", "morocco_ad_val_rate",
            "morocco_specific_rate", "morocco_other_rate", "australia_indicator",
            "australia_rate_type_code", "australia_ad_val_rate", "australia_specific_rate",
            "australia_other_rate", "bahrain_indicator", "bahrain_rate_type_code",
            "bahrain_ad_val_rate", "bahrain_specific_rate", "bahrain_other_rate",
            "dr_cafta_indicator", "dr_cafta_rate_type_code", "dr_cafta_ad_val_rate",
            "dr_cafta_specific_rate", "dr_cafta_other_rate", "dr_cafta_plus_indicator",
            "dr_cafta_plus_rate_type_code", "dr_cafta_plus_ad_val_rate",
            "dr_cafta_plus_specific_rate", "dr_cafta_plus_other_rate",
            "oman_indicator", "oman_rate_type_code", "oman_ad_val_rate",
            "oman_specific_rate", "oman_other_rate", "peru_indicator",
            "peru_rate_type_code", "peru_ad_val_rate", "peru_specific_rate", "peru_other_rate",
            "pharmaceutical_ind", "dyes_indicator", "col2_text_rate", "col2_rate_type_code",
            "col2_ad_val_rate", "col2_specific_rate", "col2_other_rate",
            "begin_effect_date", "end_effective_date", "footnote_comment", "additional_duty",
            "korea_indicator", "korea_rate_type_code", "korea_ad_val_rate",
            "korea_specific_rate", "korea_other_rate", "colombia_indicator",
            "colombia_rate_type_code", "colombia_ad_val_rate", "colombia_specific_rate",
            "colombia_other_rate", "panama_indicator", "panama_rate_type_code",
            "panama_ad_val_rate", "panama_specific_rate", "panama_other_rate",
            "nepal_indicator", "japan_indicator", "japan_rate_type_code",
            "japan_ad_val_rate", "japan_specific_rate", "japan_other_rate",
            "usmca_indicator", "usmca_rate_type_code", "usmca_ad_val_rate",
            "usmca_specific_rate", "usmca_other_rate"
        ]
        
        # Initialize all fields as None
        return {field: None for field in fields}


# Global instance
_tariff_engine = None

def get_tariff_engine() -> TariffLookupEngine:
    """Get or create the global tariff engine instance."""
    global _tariff_engine
    if _tariff_engine is None:
        _tariff_engine = TariffLookupEngine()
    return _tariff_engine
