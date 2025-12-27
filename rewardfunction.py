import json
import re
from typing import Any, Dict, List, Optional, Tuple

# ==================== Configuration ====================
DECISION_THRESHOLD = 0.85
CONF_BAND = 0.10

# ==================== JSON Extraction ====================

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from model output, handling various formats including
    reasoning text before JSON.
    """
    if not text:
        return None
    
    # If it's already a dict, return it
    if isinstance(text, dict):
        return text
    
    # Clean up the text
    text = text.strip()
    
    # Try direct JSON parse first (in case it's clean JSON)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except:
        pass
    
    # Strategy 1: Find the LAST valid JSON object in the text
    # (since model tends to output reasoning first, then JSON)
    json_objects = []
    
    # Look for JSON objects starting with {
    for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                json_objects.append(obj)
        except:
            continue
    
    # Return the last valid JSON object found
    if json_objects:
        return json_objects[-1]
    
    # Strategy 2: Use JSONDecoder to find JSON starting from any {
    decoder = json.JSONDecoder()
    positions = [m.start() for m in re.finditer(r'\{', text)]
    
    # Try from the end first (most likely to be the actual response)
    for pos in reversed(positions):
        try:
            obj, _ = decoder.raw_decode(text[pos:])
            if isinstance(obj, dict):
                # Validate it has expected fields for any task
                expected_fields = [
                    'chapters', 'selected_indices', 'option_number',
                    'question_type', 'updated_description', 'suggested_code'
                ]
                if any(field in obj for field in expected_fields):
                    return obj
        except:
            continue
    
    return None


def get_user_data(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract the user's JSON payload from messages."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return extract_json_from_text(content)
    return None


def get_assistant_data(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract the assistant's JSON response from messages."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return extract_json_from_text(content)
    return None


# ==================== Confidence Band Logic ====================

def calculate_confidence_band(target_conf: float, threshold: float) -> Tuple[float, float]:
    """Calculate the allowed confidence band based on target and decision threshold."""
    if target_conf >= threshold:
        low = threshold
        high = min(1.0, target_conf + CONF_BAND)
    else:
        low = max(0.0, target_conf - CONF_BAND)
        high = min(max(0.0, threshold - 0.01), target_conf + CONF_BAND)
    return low, high


def hinge_gate_score(pred_conf: float, low: float, high: float) -> float:
    """Smooth gate: 1.0 inside [low, high], linear decay outside."""
    if low <= pred_conf <= high:
        return 1.0
    distance = max(0.0, low - pred_conf, pred_conf - high)
    band_width = max(0.01, high - low)
    return max(0.0, 1.0 - distance / band_width)


def closeness_score(pred_conf: float, target_conf: float, low: float, high: float) -> float:
    """Proximity to target within the allowed band. 0 if outside."""
    if pred_conf < low or pred_conf > high:
        return 0.0
    max_distance = max(target_conf - low, high - target_conf, 0.01)
    return max(0.0, 1.0 - abs(pred_conf - target_conf) / max_distance)


# ==================== Task Evaluators ====================

def evaluate_select_chapters(user_data: Dict, assistant_data: Dict, targets: Dict, threshold: float) -> Dict:
    """Evaluate chapter selection."""
    tgt_list = targets.get("chapters", [])
    if not isinstance(tgt_list, list) or not tgt_list:
        return {"score": 0.0, "is_score_valid": False, "reason": "Missing targets.chapters"}

    gt_chapter = str(tgt_list[0].get("chapter", "")).strip()
    if not gt_chapter:
        return {"score": 0.0, "is_score_valid": False, "reason": "Invalid gold chapter"}

    gold_conf_map: Dict[str, float] = {}
    for i, e in enumerate(tgt_list):
        ch = str(e.get("chapter", "")).strip()
        conf = e.get("confidence")
        if not ch or not isinstance(conf, (int, float)):
            return {"score": 0.0, "is_score_valid": False, "reason": f"Invalid targets.chapters entry at {i}"}
        gold_conf_map[ch] = float(conf)

    pred_list = assistant_data.get("chapters", [])
    if not isinstance(pred_list, list) or not pred_list:
        return {"score": 0.0, "is_score_valid": False, "reason": "Missing 'chapters' in assistant response"}

    pred_pairs: List[Tuple[str, float]] = []
    for i, e in enumerate(pred_list):
        ch = str(e.get("chapter", "")).strip()
        conf = e.get("confidence")
        if not ch or not isinstance(conf, (int, float)):
            return {"score": 0.0, "is_score_valid": False, "reason": f"Invalid assistant chapters entry at {i}"}
        pred_pairs.append((ch, float(conf)))

    pred_order = [c for c, _ in pred_pairs]
    if gt_chapter not in pred_order:
        return {"score": 0.0, "is_score_valid": True, "reason": f"GT chapter {gt_chapter} not in predictions"}

    ranked = sorted(pred_pairs, key=lambda x: -x[1])
    ranked_nums = [c for c, _ in ranked]
    rank = ranked_nums.index(gt_chapter) + 1
    pred_conf_for_gt = dict(pred_pairs).get(gt_chapter, 0.0)
    target_conf_for_gt = gold_conf_map.get(gt_chapter, 0.0)

    top1 = 1.0 if rank == 1 else 0.0
    recip_rank = 1.0 / rank

    low, high = calculate_confidence_band(target_conf_for_gt, threshold)
    gate = hinge_gate_score(pred_conf_for_gt, low, high)
    close = closeness_score(pred_conf_for_gt, target_conf_for_gt, low, high)

    score = min(1.0, 0.70 * top1 + 0.15 * recip_rank + 0.10 * gate + 0.05 * close)

    return {
        "score": score,
        "is_score_valid": True,
        "reason": f"GT {gt_chapter} rank #{rank}; pred_conf={pred_conf_for_gt:.2f}, target_conf={target_conf_for_gt:.2f}"
    }


def evaluate_select_candidates(user_data: Dict, assistant_data: Dict, targets: Dict, threshold: float) -> Dict:
    """Evaluate candidate selection."""
    gold_sel = targets.get("selected_indices", [])
    if not isinstance(gold_sel, list) or not gold_sel:
        return {"score": 0.0, "is_score_valid": False, "reason": "Missing targets.selected_indices"}

    try:
        gt_index = int(gold_sel[0])
    except:
        return {"score": 0.0, "is_score_valid": False, "reason": "Invalid gold index"}

    selected = assistant_data.get("selected_indices", [])
    if not isinstance(selected, list) or not selected:
        return {"score": 0.0, "is_score_valid": False, "reason": "Missing 'selected_indices' in assistant response"}

    try:
        pred_sel = [int(x) for x in selected]
    except:
        return {"score": 0.0, "is_score_valid": False, "reason": "Non-integer index in assistant selected_indices"}

    if gt_index not in pred_sel:
        return {"score": 0.0, "is_score_valid": True, "reason": f"GT index {gt_index} not selected"}

    rank = pred_sel.index(gt_index) + 1
    top1 = 1.0 if rank == 1 else 0.0
    recip_rank = 1.0 / rank
    score = min(1.0, 0.85 * top1 + 0.15 * recip_rank)

    return {"score": score, "is_score_valid": True, "reason": f"Index {gt_index} ranked #{rank}"}


def evaluate_score_candidate(user_data: Dict, assistant_data: Dict, targets: Dict, threshold: float) -> Dict:
    """Evaluate single candidate scoring."""
    cand = user_data.get("data", {}).get("candidate", {})
    cand_index = cand.get("index")
    if cand_index is None:
        return {"score": 0.0, "is_score_valid": False, "reason": "Missing candidate index"}

    try:
        gold_option = int(targets.get("option_number"))
        gold_conf = float(targets.get("confidence"))
    except:
        return {"score": 0.0, "is_score_valid": False, "reason": "Invalid targets"}

    option_num = assistant_data.get("option_number")
    pred_conf = assistant_data.get("confidence")
    if option_num is None or pred_conf is None:
        return {"score": 0.0, "is_score_valid": False, "reason": "Missing option_number or confidence"}

    try:
        option_num = int(option_num)
        pred_conf = float(pred_conf)
    except:
        return {"score": 0.0, "is_score_valid": False, "reason": "Non-numeric values"}

    if option_num != cand_index:
        return {"score": 0.0, "is_score_valid": True, "reason": f"Wrong candidate: {option_num} != {cand_index}"}

    if option_num != gold_option:
        return {"score": 0.0, "is_score_valid": True, "reason": f"Index mismatch: {option_num} != {gold_option}"}

    low, high = calculate_confidence_band(gold_conf, threshold)
    gate = hinge_gate_score(pred_conf, low, high)
    close = closeness_score(pred_conf, gold_conf, low, high)

    score = min(1.0, 0.70 * gate + 0.30 * close)

    return {
        "score": score,
        "is_score_valid": True,
        "reason": f"pred_conf={pred_conf:.2f}, target_conf={gold_conf:.2f}"
    }


# ==================== Main Entry Point ====================

def evaluate(messages: List[Dict[str, Any]], targets: Dict) -> Dict:
    """Evaluate HTS classification model responses."""
    try:
        user_data = get_user_data(messages)
        if not user_data:
            return {"score": 0.0, "is_score_valid": False, "reason": "Failed to parse user JSON"}

        assistant_data = get_assistant_data(messages)
        if not assistant_data:
            return {"score": 0.0, "is_score_valid": False, "reason": "No assistant JSON to evaluate"}

        task = user_data.get("task")
        if not task:
            return {"score": 0.0, "is_score_valid": False, "reason": "Missing 'task' in user data"}

        threshold = float(user_data.get("data", {}).get("confidence_threshold", DECISION_THRESHOLD))

        if task == "select_chapters":
            result = evaluate_select_chapters(user_data, assistant_data, targets, threshold)
        elif task == "select_candidates":
            result = evaluate_select_candidates(user_data, assistant_data, targets, threshold)
        elif task == "score_candidate":
            result = evaluate_score_candidate(user_data, assistant_data, targets, threshold)
        else:
            result = {"score": 0.0, "is_score_valid": False, "reason": f"Unknown task: {task}"}

        return {
            "score": float(result.get("score", 0.0)),
            "is_score_valid": bool(result.get("is_score_valid", False)),
            "reason": result.get("reason", "Unknown error")
        }

    except Exception as e:
        return {"score": 0.0, "is_score_valid": False, "reason": f"Error: {str(e)}"}