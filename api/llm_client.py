import json
import logging
import os
import threading
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None
    GROQ_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

try:
    from anthropic import AnthropicVertex
    ANTHROPIC_AVAILABLE = True
except ImportError:
    AnthropicVertex = None
    ANTHROPIC_AVAILABLE = False

from .training_data_collector import training_collector, build_state
from .system_prompts_updated import UNIFIED_SYSTEM_PROMPT

# --- Custom vLLM constants (hard-coded as requested) ---
CUSTOM_VLLM_BASE_URL = "http://86.38.238.198:8000/v1"
CUSTOM_VLLM_MODEL = "openai/gpt-oss-120b"
CUSTOM_VLLM_TIMEOUT = 120.0
CUSTOM_VLLM_MAX_TOKENS = 10000

# --- Optional OpenAI and Groq settings (env overrides still supported) ---
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-5.2")
DEFAULT_OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "4096"))
DEFAULT_OPENAI_TIMEOUT = float(os.environ.get("OPENAI_TIMEOUT", "300"))  # 5 minutes for GPT-5.1 reasoning

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
DEFAULT_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "medium")
DEFAULT_VERBOSITY = os.environ.get("OPENAI_VERBOSITY", "medium")

OPENROUTER_API_URL = os.environ.get("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-5.2")
OPENROUTER_MAX_TOKENS = int(os.environ.get("OPENROUTER_MAX_TOKENS", "65536"))
OPENROUTER_TIMEOUT = float(os.environ.get("OPENROUTER_TIMEOUT", "600"))  # 10 min for GPT-5.1 reasoning
OPENROUTER_REASONING_EFFORT = os.environ.get("OPENROUTER_REASONING_EFFORT", "medium")
OPENROUTER_TEMPERATURE = float(os.environ.get("OPENROUTER_TEMPERATURE", "0"))
OPENROUTER_REFERER = os.environ.get("OPENROUTER_REFERER", "")
OPENROUTER_TITLE = os.environ.get("OPENROUTER_TITLE", "HSCode Trainer")

# --- Gemini API settings (OpenAI-compatible endpoint) ---
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_MAX_TOKENS = int(os.environ.get("GEMINI_MAX_TOKENS", "8192"))
GEMINI_TIMEOUT = float(os.environ.get("GEMINI_TIMEOUT", "300"))
GEMINI_REASONING_EFFORT = os.environ.get("GEMINI_REASONING_EFFORT", "high")  # high = 24,576 thinking budget

# --- Anthropic Vertex AI settings ---
ANTHROPIC_REGION = os.environ.get("ANTHROPIC_REGION", "global")
ANTHROPIC_PROJECT_ID = os.environ.get("ANTHROPIC_PROJECT_ID", os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-5@20251101")
ANTHROPIC_MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "8192"))

# OpenRouter provider filtering: https://openrouter.ai/docs/provider-routing
OPENROUTER_PROVIDER_ORDER = [
    provider.strip()
    for provider in os.environ.get("OPENROUTER_PROVIDER_ORDER", "").split(",")
    if provider.strip()
]
OPENROUTER_PROVIDER_ONLY = [
    provider.strip()
    for provider in os.environ.get("OPENROUTER_PROVIDER_ONLY", "").split(",")
    if provider.strip()
]
OPENROUTER_PROVIDER_IGNORE = [
    provider.strip()
    for provider in os.environ.get("OPENROUTER_PROVIDER_IGNORE", "").split(",")
    if provider.strip()
]

JSON_REQUIREMENTS = (
    "CRITICAL JSON REQUIREMENTS:\n"
    "- Return ONLY valid JSON with no additional text\n"
    "- Do not include any explanation before or after the JSON\n"
    "- If generating an array, ensure it starts with [ and ends with ]\n"
    "- All JSON objects must be properly separated by commas within the array\n"
    "- Do not generate separate JSON objects - they must be inside an array"
)


class LLMClient:
    """
    Centralized LLM adapter for the tree engine.

    - Defaults to a self-hosted vLLM node that speaks the OpenAI Chat Completions API.
    - Supports fallbacks to the regular OpenAI platform and Groq.
    - Keeps the legacy public interface so existing engines keep working.
    """

    def __init__(self):
        self.system_prompt = UNIFIED_SYSTEM_PROMPT.strip()
        self.log_prompts = os.environ.get("LOG_PROMPTS", "false").lower() == "true"
        self.default_provider = self._determine_default_provider()
        self.retry_attempts = max(1, int(os.environ.get("LLM_RETRY_ATTEMPTS", "3")))

        self._tl = threading.local()
        self._tl.system_prompt_injection = None

        self._vllm_client: Optional[Any] = None
        self._openai_client: Optional[Any] = None
        self._gemini_client: Optional[Any] = None
        self._anthropic_client: Optional[Any] = None
        self._openrouter_session: Optional[requests.Session] = None
        self.client: Optional[Any] = self._maybe_create_groq_client()

        if self.log_prompts:
            self._setup_prompt_logger()

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def set_system_prompt_injection(self, prompt: Optional[str]) -> None:
        """Allow worker threads to inject a temporary system prompt."""
        self._tl.system_prompt_injection = prompt

    def clear_system_prompt_injection(self) -> None:
        """Remove any previously injected system prompt."""
        if hasattr(self._tl, "system_prompt_injection"):
            self._tl.system_prompt_injection = None

    def _current_system_prompt(self) -> Tuple[str, bool]:
        """Return the active system prompt and whether it was injected."""
        override = getattr(self._tl, "system_prompt_injection", None)
        if override:
            return override, True
        return self.system_prompt, False

    def _build_messages(self, prompt: str, requires_json: bool) -> Tuple[List[Dict[str, str]], str, str, bool]:
        """Assemble the OpenAI-style message payload."""
        system_prompt, injected = self._current_system_prompt()
        user_content = prompt.strip()
        if requires_json:
            user_content = f"{user_content.rstrip()}\n\n{JSON_REQUIREMENTS}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return messages, user_content, system_prompt, injected

    def _stringify_content_parts(self, data: Any) -> str:
        """Convert OpenAI SDK content parts into a single string."""
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        if isinstance(data, list):
            parts: List[str] = []
            for item in data:
                text = ""
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict):
                    if "text" in item:
                        text = self._stringify_content_parts(item.get("text"))
                    elif "content" in item:
                        text = self._stringify_content_parts(item.get("content"))
                    elif "output_text" in item:
                        text = self._stringify_content_parts(item.get("output_text"))
                    elif "value" in item:
                        text = self._stringify_content_parts(item.get("value"))
                    elif "data" in item:
                        text = self._stringify_content_parts(item.get("data"))
                else:
                    text_attr = getattr(item, "text", None)
                    if text_attr:
                        text = self._stringify_content_parts(text_attr)
                    else:
                        content_attr = getattr(item, "content", None)
                        if content_attr:
                            text = self._stringify_content_parts(content_attr)
                if text:
                    parts.append(text)
            return "".join(parts)
        if isinstance(data, dict):
            for key in ("text", "content", "output_text", "value"):
                if key in data:
                    text = self._stringify_content_parts(data[key])
                    if text:
                        return text
            try:
                return json.dumps(data, ensure_ascii=False)
            except Exception:
                return str(data)
        return str(data)

    def _stringify_structured_payload(self, payload: Any) -> str:
        """Stringify structured payloads returned by newer chat completions."""
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return str(payload)

    def _extract_choice_content(self, choice: Any, requires_json: bool) -> str:
        """Safely extract assistant text from a ChatCompletion choice."""
        if not choice:
            return ""

        message = getattr(choice, "message", None)
        if message:
            content = self._stringify_content_parts(getattr(message, "content", None))
            if content.strip():
                return content

            # Structured output fields introduced for GPT-5 reasoning
            parsed = getattr(message, "parsed", None)
            if parsed is not None:
                parsed_text = self._stringify_structured_payload(parsed)
                if parsed_text.strip():
                    return parsed_text

            output = getattr(message, "output", None)
            if output is not None:
                output_text = self._stringify_content_parts(output)
                if output_text.strip():
                    return output_text

            if requires_json:
                for attr in ("tool_calls", "function_call"):
                    if hasattr(message, attr):
                        structured = getattr(message, attr)
                        structured_text = self._stringify_structured_payload(structured)
                        if structured_text.strip():
                            return structured_text

            refusal = getattr(message, "refusal", None)
            if refusal:
                return refusal

        # Legacy responses may use choice.text
        text_choice = getattr(choice, "text", None)
        if isinstance(text_choice, str) and text_choice.strip():
            return text_choice

        return ""

    def _setup_prompt_logger(self) -> None:
        """Optional file logger for auditing prompts."""
        prompt_log_file = os.environ.get("PROMPT_LOG_FILE", "groq_prompts.log")
        prompt_log_dir = os.path.dirname(prompt_log_file)
        if prompt_log_dir and not os.path.exists(prompt_log_dir):
            os.makedirs(prompt_log_dir, exist_ok=True)

        self.prompt_logger = logging.getLogger("llm_prompts")
        self.prompt_logger.setLevel(logging.INFO)
        self.prompt_logger.handlers.clear()

        handler = logging.FileHandler(prompt_log_file, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.prompt_logger.addHandler(handler)
        self.prompt_logger.propagate = False

    # ------------------------------------------------------------------
    # Client factories
    # ------------------------------------------------------------------

    def _maybe_create_groq_client(self) -> Optional[Any]:
        if not GROQ_AVAILABLE:
            logging.info("Groq SDK not installed; Groq support disabled.")
            return None
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logging.info("GROQ_API_KEY not set; Groq support disabled.")
            return None
        try:
            return Groq(api_key=api_key)
        except Exception as exc:
            logging.error(f"Failed to initialize Groq client: {exc}")
            return None

    def _get_vllm_client(self):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI SDK is required for the custom vLLM endpoint.")
        if self._vllm_client is None:
            self._vllm_client = OpenAI(
                api_key="local-vllm",
                base_url=CUSTOM_VLLM_BASE_URL,
                timeout=CUSTOM_VLLM_TIMEOUT,
            )
        return self._vllm_client

    def _get_openai_client(self):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI SDK is not installed. Run `pip install openai`.")
        if self._openai_client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable is required for OpenAI fallback.")
            self._openai_client = OpenAI(api_key=api_key, timeout=DEFAULT_OPENAI_TIMEOUT)
        return self._openai_client

    def _get_gemini_client(self):
        """Get or create the Gemini client using OpenAI-compatible endpoint."""
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI SDK is required for Gemini API (OpenAI-compatible mode).")
        if self._gemini_client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable is required for Gemini provider.")
            self._gemini_client = OpenAI(
                api_key=api_key,
                base_url=GEMINI_API_BASE_URL,
                timeout=GEMINI_TIMEOUT,
            )
        return self._gemini_client

    def _get_anthropic_client(self):
        """Get or create the Anthropic Vertex AI client."""
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("Anthropic SDK is not installed. Run `pip install anthropic`.")
        if self._anthropic_client is None:
            if not ANTHROPIC_PROJECT_ID:
                raise RuntimeError("ANTHROPIC_PROJECT_ID (or GOOGLE_CLOUD_PROJECT) environment variable is required for Anthropic Vertex provider.")
            self._anthropic_client = AnthropicVertex(
                region=ANTHROPIC_REGION,
                project_id=ANTHROPIC_PROJECT_ID,
            )
            logging.info(f"Initialized AnthropicVertex client (region={ANTHROPIC_REGION}, project={ANTHROPIC_PROJECT_ID})")
        return self._anthropic_client

    def _get_openrouter_session(self) -> requests.Session:
        if self._openrouter_session is None:
            self._openrouter_session = requests.Session()
        return self._openrouter_session

    def _determine_default_provider(self) -> str:
        override = os.environ.get("LLM_PROVIDER")
        if override:
            return override.strip().lower()
        # Default to OpenRouter (GPT-5.2 with medium reasoning)
        if os.environ.get("OPENROUTER_API_KEY"):
            return "openrouter"
        # Fallback to Gemini 3 Flash
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        # Fallback to Anthropic Vertex if available and project ID is set
        if ANTHROPIC_AVAILABLE and ANTHROPIC_PROJECT_ID:
            return "anthropic"
        if GROQ_AVAILABLE and os.environ.get("GROQ_API_KEY"):
            return "groq"
        return "openrouter"

    def _resolve_provider(self, override: Optional[str]) -> str:
        provider = (override or self.default_provider).strip().lower()
        if provider not in {"vllm", "openai", "openrouter", "groq", "gemini", "anthropic"}:
            provider = self.default_provider
        return provider

    # ------------------------------------------------------------------
    # Core chat completions
    # ------------------------------------------------------------------

    def _call_chat_completion(
        self,
        provider: str,
        messages: List[Dict[str, str]],
        requires_json: bool,
        temperature: float,
    ) -> Tuple[str, str]:
        if provider == "anthropic":
            return self._call_anthropic_chat_completion(
                messages=messages,
                requires_json=requires_json,
                temperature=temperature,
            )
        if provider == "openrouter":
            return self._call_openrouter_chat_completion(
                messages=messages,
                requires_json=requires_json,
                temperature=temperature,
            )
        if provider == "gemini":
            return self._call_gemini_chat_completion(
                messages=messages,
                requires_json=requires_json,
                temperature=temperature,
            )
        if provider == "groq":
            groq_client = self.client or self._maybe_create_groq_client()
            if groq_client is None:
                raise RuntimeError("Groq client is not configured. Set GROQ_API_KEY to enable it.")
            kwargs = {
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 8192,
            }
            completion = groq_client.chat.completions.create(**kwargs)
            choice = completion.choices[0]
            content = self._extract_choice_content(choice, requires_json).strip()
            if not content:
                raise RuntimeError("Empty Groq chat response.")
            return content, GROQ_MODEL
        if provider == "vllm":
            client = self._get_vllm_client()
            model = CUSTOM_VLLM_MODEL
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": CUSTOM_VLLM_MAX_TOKENS,
            }
        else:
            client = self._get_openai_client()
            model = DEFAULT_OPENAI_MODEL
            kwargs = {
                "model": model,
                "messages": messages,
                # Note: GPT-5.1 doesn't support custom temperature, only default (1)
                "max_completion_tokens": DEFAULT_OPENAI_MAX_TOKENS,  # gpt-5 uses max_completion_tokens
                "store": False,  # Don't store for privacy
            }
            # Set response format (gpt-5.1 requires explicit format)
            if requires_json:
                kwargs["response_format"] = {"type": "json_object"}
            else:
                kwargs["response_format"] = {"type": "text"}
            # Add reasoning parameters for gpt-5.1
            if DEFAULT_REASONING_EFFORT:
                kwargs["reasoning_effort"] = DEFAULT_REASONING_EFFORT
            if DEFAULT_VERBOSITY:
                kwargs["verbosity"] = DEFAULT_VERBOSITY

        completion = client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        content = self._extract_choice_content(choice, requires_json).strip()
        if not content:
            logging.debug("Chat completion response had no textual content: %s", completion)
            raise RuntimeError("Empty chat completion response.")
        return content, model

    def _call_openrouter_chat_completion(
        self,
        messages: List[Dict[str, str]],
        requires_json: bool,
        temperature: float,
    ) -> Tuple[str, str]:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable is required for OpenRouter provider.")

        payload: Dict[str, Any] = {
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "temperature": OPENROUTER_TEMPERATURE,  # Use configured temp (default 0 for determinism)
            "max_tokens": OPENROUTER_MAX_TOKENS,
        }
        
        # Handle response format and reasoning based on model
        is_kimi_k2 = "kimi-k2" in OPENROUTER_MODEL.lower()
        
        if requires_json:
            payload["response_format"] = {"type": "json_object"}
        
        # Kimi K2 thinking models require reasoning in extra_body
        if is_kimi_k2:
            payload["extra_body"] = {"reasoning": {"enabled": True}}
        elif OPENROUTER_REASONING_EFFORT:
            # Other reasoning models use top-level reasoning parameter
            payload["reasoning"] = {"effort": OPENROUTER_REASONING_EFFORT}
        
        # OpenRouter provider routing (proper API fields per docs)
        provider_config: Dict[str, Any] = {}
        if OPENROUTER_PROVIDER_ORDER:
            provider_config["order"] = OPENROUTER_PROVIDER_ORDER
        if OPENROUTER_PROVIDER_ONLY:
            provider_config["only"] = OPENROUTER_PROVIDER_ONLY
        if OPENROUTER_PROVIDER_IGNORE:
            provider_config["ignore"] = OPENROUTER_PROVIDER_IGNORE
        if provider_config:
            payload["provider"] = provider_config

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if OPENROUTER_REFERER:
            headers["HTTP-Referer"] = OPENROUTER_REFERER
        if OPENROUTER_TITLE:
            headers["X-Title"] = OPENROUTER_TITLE

        session = self._get_openrouter_session()
        try:
            response = session.post(
                OPENROUTER_API_URL,
                json=payload,
                headers=headers,
                timeout=OPENROUTER_TIMEOUT,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            error_detail = ""
            try:
                if hasattr(exc, 'response') and exc.response is not None:
                    error_detail = f" Response: {exc.response.text[:500]}"
            except Exception:
                pass
            raise RuntimeError(f"OpenRouter API request failed: {exc}{error_detail}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"OpenRouter returned invalid JSON: {response.text[:200]}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter response did not contain choices: {data}")

        message = choices[0].get("message", {}) or {}
        
        # Extract content (prefer main content over reasoning)
        content_text = self._stringify_content_parts(message.get("content"))
        
        # For Kimi K2, reasoning output might be in a separate field
        if not content_text:
            reasoning_content = message.get("reasoning")
            if reasoning_content:
                content_text = self._stringify_content_parts(reasoning_content)
        
        if not content_text and requires_json:
            for structured_key in ("tool_calls", "function_call"):
                if structured_key in message and message[structured_key]:
                    content_text = self._stringify_structured_payload(message[structured_key])
                    if content_text:
                        break
        if not content_text:
            content_text = (message.get("refusal") or "").strip()

        content_text = (content_text or "").strip()
        if not content_text:
            raise RuntimeError(f"OpenRouter response did not include content: {data}")

        model_name = data.get("model") or OPENROUTER_MODEL
        return content_text, model_name

    def _call_gemini_chat_completion(
        self,
        messages: List[Dict[str, str]],
        requires_json: bool,
        temperature: float,
    ) -> Tuple[str, str]:
        """Call Gemini API using OpenAI-compatible endpoint with reasoning support."""
        client = self._get_gemini_client()
        
        kwargs: Dict[str, Any] = {
            "model": GEMINI_MODEL,
            "messages": messages,
            "max_tokens": GEMINI_MAX_TOKENS,
        }
        
        # Gemini 3 models support reasoning_effort: "high" = 24,576 thinking budget
        if GEMINI_REASONING_EFFORT:
            kwargs["reasoning_effort"] = GEMINI_REASONING_EFFORT
        
        # Set temperature (user override - Google recommends temp=1.0 for Gemini 3 but we're testing temp=0)
        kwargs["temperature"] = temperature
        
        # Set response format for JSON if needed
        if requires_json:
            kwargs["response_format"] = {"type": "json_object"}
        
        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Provide helpful error message
            error_str = str(exc)
            if "API key" in error_str or "authentication" in error_str.lower():
                raise RuntimeError(f"Gemini API authentication failed. Check your GEMINI_API_KEY: {exc}") from exc
            raise RuntimeError(f"Gemini API call failed: {exc}") from exc
        
        choice = completion.choices[0] if completion.choices else None
        if not choice:
            raise RuntimeError("Gemini response did not contain any choices.")
        
        content = self._extract_choice_content(choice, requires_json).strip()
        if not content:
            logging.debug("Gemini chat completion response had no textual content: %s", completion)
            raise RuntimeError("Empty Gemini chat completion response.")
        
        return content, GEMINI_MODEL

    def _call_anthropic_chat_completion(
        self,
        messages: List[Dict[str, str]],
        requires_json: bool,
        temperature: float,
    ) -> Tuple[str, str]:
        """Call Anthropic API via Vertex AI using native SDK."""
        client = self._get_anthropic_client()
        
        # Anthropic uses a different message format - extract system prompt
        system_content = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)
        
        # Build kwargs for Anthropic
        kwargs: Dict[str, Any] = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "messages": user_messages,
        }
        
        # Add system prompt if present
        if system_content:
            kwargs["system"] = system_content
        
        # Set temperature (Anthropic supports 0-1)
        if temperature >= 0:
            kwargs["temperature"] = temperature
        
        try:
            message = client.messages.create(**kwargs)
        except Exception as exc:
            error_str = str(exc)
            if "authentication" in error_str.lower() or "permission" in error_str.lower():
                raise RuntimeError(f"Anthropic Vertex authentication failed. Check your GCP credentials: {exc}") from exc
            raise RuntimeError(f"Anthropic API call failed: {exc}") from exc
        
        # Extract content from Anthropic response
        if not message.content:
            raise RuntimeError("Anthropic response did not contain any content.")
        
        # Anthropic returns content as a list of content blocks
        content_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                content_text += block.text
            elif isinstance(block, dict) and "text" in block:
                content_text += block["text"]
        
        content_text = content_text.strip()
        if not content_text:
            logging.debug("Anthropic chat completion response had no textual content: %s", message)
            raise RuntimeError("Empty Anthropic chat completion response.")
        
        return content_text, ANTHROPIC_MODEL

    # ------------------------------------------------------------------
    # Public API used by the rest of the codebase
    # ------------------------------------------------------------------

    def send_openai_request(
        self,
        prompt: str,
        requires_json: bool = False,
        temperature: float = 0.0,
        task_type: Optional[str] = None,
        product_text: Optional[str] = None,
        research: Optional[str] = None,  # kept for signature compatibility
        prompt_json: Optional[Dict[str, Any]] = None,
        enable_web_search: bool = False,  # unused but preserved for compatibility
        provider_override: Optional[str] = None,
        **training_kwargs: Any,
    ) -> str:
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI SDK not available. Install with `pip install openai`.")

        if product_text and "product_text" not in training_kwargs:
            training_kwargs["product_text"] = product_text

        provider = self._resolve_provider(provider_override)
        messages, user_prompt, system_prompt, injected = self._build_messages(prompt, requires_json)
        system_preview = messages[0]["content"] if messages else ""

        if injected:
            inj_hash = hashlib.md5(system_preview.encode("utf-8")).hexdigest()[:8]
            logging.info(f"Using system prompt injection (hash={inj_hash})")

        if prompt_json is None and task_type:
            try:
                prompt_json = build_state(task_type, **training_kwargs)
            except Exception as exc:
                logging.error(f"Failed to build training state: {exc}")
                prompt_json = {"task": task_type, "raw_prompt": prompt}

        attempt = 0
        final_text: Optional[str] = None
        model_used = ""
        last_error: Optional[Exception] = None

        while attempt < self.retry_attempts:
            attempt += 1
            try:
                response_text, model_used = self._call_chat_completion(
                    provider=provider,
                    messages=messages,
                    requires_json=requires_json,
                    temperature=temperature,
                )
                if requires_json:
                    cleaned = self._extract_json_from_response(response_text)
                    json.loads(cleaned)
                    final_text = cleaned
                else:
                    final_text = response_text
                break
            except Exception as exc:
                last_error = exc
                logging.warning(
                    f"LLM request attempt {attempt}/{self.retry_attempts} failed ({provider}): {exc}"
                )
                if attempt < self.retry_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))

        if final_text is None:
            raise RuntimeError(f"LLM request failed after {self.retry_attempts} attempts: {last_error}")

        if self.log_prompts:
            try:
                self.prompt_logger.info(
                    f"==== CHAT COMPLETION ({provider}, model: {model_used}) ====\n"
                    f"System Prompt Preview: {system_preview[:200]}...\n"
                    f"User Prompt:\n{user_prompt}\n"
                    f"Requires JSON: {requires_json}\n"
                    f"Temperature: {temperature}\n"
                    f"==== RESPONSE ====\n{final_text}\n==== END ===="
                )
            except Exception as log_exc:
                logging.warning(f"Failed to log prompt: {log_exc}")

        if training_collector.enabled and prompt_json and task_type:
            try:
                training_collector.log_request(
                    task_type=task_type,
                    prompt_json=prompt_json,
                    response=final_text,
                    metadata={
                        "model": model_used,
                        "provider": provider,
                        "temperature": temperature,
                        "requires_json": requires_json,
                    },
                    system_prompt=system_prompt,
                )
            except Exception as exc:
                logging.error(f"Failed to log training data: {exc}")

        logging.info(f"LLM call succeeded using provider '{provider}' and model '{model_used}'")
        return final_text

    def send_trajectory_request(
        self,
        messages: List[Dict[str, str]],
        requires_json: bool = False,
        temperature: float = 0.0,
        task_type: Optional[str] = None,
        prompt_json: Optional[Dict[str, Any]] = None,
        provider_override: Optional[str] = None,
    ) -> str:
        """
        Send a request using a pre-built message trajectory.
        
        This method accepts a complete list of messages (system + prior turns + new user message)
        and sends it to the LLM. Unlike send_openai_request, it does NOT build messages from
        scratch - it uses the provided trajectory directly.
        
        Args:
            messages: Complete message list in OpenAI format 
                      [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
            requires_json: Whether response should be valid JSON
            temperature: Sampling temperature
            task_type: Optional task type for logging/training
            prompt_json: Optional structured prompt for training data collection
            provider_override: Override the default LLM provider
            
        Returns:
            The assistant's response text
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        provider = self._resolve_provider(provider_override)
        
        # Make a copy of messages to avoid mutating the original
        request_messages = [msg.copy() for msg in messages]
        
        # Add JSON requirements to the last user message if needed
        if requires_json and request_messages:
            # Find the last user message
            for i in range(len(request_messages) - 1, -1, -1):
                if request_messages[i]["role"] == "user":
                    request_messages[i]["content"] = f"{request_messages[i]['content'].rstrip()}\n\n{JSON_REQUIREMENTS}"
                    break
        
        # Extract system prompt for logging
        system_preview = ""
        user_preview = ""
        for msg in request_messages:
            if msg["role"] == "system" and not system_preview:
                system_preview = msg["content"][:200]
            if msg["role"] == "user":
                user_preview = msg["content"]  # Keep last user message
        
        attempt = 0
        final_text: Optional[str] = None
        model_used = ""
        last_error: Optional[Exception] = None
        
        while attempt < self.retry_attempts:
            attempt += 1
            try:
                response_text, model_used = self._call_chat_completion(
                    provider=provider,
                    messages=request_messages,
                    requires_json=requires_json,
                    temperature=temperature,
                )
                if requires_json:
                    cleaned = self._extract_json_from_response(response_text)
                    json.loads(cleaned)  # Validate JSON
                    final_text = cleaned
                else:
                    final_text = response_text
                break
            except Exception as exc:
                last_error = exc
                logging.warning(
                    f"Trajectory LLM request attempt {attempt}/{self.retry_attempts} failed ({provider}): {exc}"
                )
                if attempt < self.retry_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
        
        if final_text is None:
            raise RuntimeError(f"Trajectory LLM request failed after {self.retry_attempts} attempts: {last_error}")
        
        if self.log_prompts:
            try:
                self.prompt_logger.info(
                    f"==== TRAJECTORY CHAT COMPLETION ({provider}, model: {model_used}) ====\n"
                    f"Messages count: {len(request_messages)}\n"
                    f"System Prompt Preview: {system_preview}...\n"
                    f"Last User Message:\n{user_preview[:500]}...\n"
                    f"Requires JSON: {requires_json}\n"
                    f"Temperature: {temperature}\n"
                    f"==== RESPONSE ====\n{final_text}\n==== END ===="
                )
            except Exception as log_exc:
                logging.warning(f"Failed to log trajectory prompt: {log_exc}")
        
        # Training data collection - pass FULL trajectory for multi-turn training
        if training_collector.enabled and prompt_json and task_type:
            try:
                # Pass the full trajectory (system + all prior turns + current user message)
                # The training collector will append the assistant response
                training_collector.log_request(
                    task_type=task_type,
                    prompt_json=prompt_json,
                    response=final_text,
                    metadata={
                        "model": model_used,
                        "provider": provider,
                        "temperature": temperature,
                        "requires_json": requires_json,
                        "trajectory_mode": True,
                        "message_count": len(request_messages),
                    },
                    trajectory_messages=request_messages,  # Full trajectory for multi-turn training
                )
            except Exception as exc:
                logging.error(f"Failed to log trajectory training data: {exc}")
        
        logging.info(f"Trajectory LLM call succeeded using provider '{provider}' and model '{model_used}' ({len(messages)} messages)")
        return final_text

    def send_groq_request(self, prompt: str, requires_json: bool = False, temperature: float = 0.0) -> str:
        """Direct access to Groq chat completions (used by legacy CLI)."""
        client = self.client or self._maybe_create_groq_client()
        if client is None:
            raise RuntimeError("Groq client is not configured. Set GROQ_API_KEY to enable it.")

        user_prompt = prompt
        if requires_json:
            user_prompt = f"{prompt.rstrip()}\n\n{JSON_REQUIREMENTS}"

        attempt = 0
        last_error: Optional[Exception] = None
        response_text: Optional[str] = None

        while attempt < self.retry_attempts:
            attempt += 1
            try:
                completion = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=temperature,
                    max_tokens=8192,
                )
                response_text = completion.choices[0].message.content.strip()
                if not response_text:
                    raise RuntimeError("Empty Groq response.")
                if requires_json:
                    cleaned = self._extract_json_from_response(response_text)
                    json.loads(cleaned)
                    response_text = cleaned
                break
            except Exception as exc:
                last_error = exc
                logging.warning(
                    f"Groq request attempt {attempt}/{self.retry_attempts} failed: {exc}"
                )
                if attempt < self.retry_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))

        if response_text is None:
            raise RuntimeError(f"Groq request failed after {self.retry_attempts} attempts: {last_error}")
        return response_text

    # Backward-compatibility alias
    def send_vertex_ai_request(self, *args, **kwargs) -> str:
        """Vertex AI support was removed; reuse the OpenAI/vLLM path instead."""
        return self.send_openai_request(*args, **kwargs)

    # ------------------------------------------------------------------
    # JSON extraction helper (still used across the tree engine)
    # ------------------------------------------------------------------

    def _extract_json_from_response(self, response_text: str) -> str:
        if not response_text:
            raise ValueError("No response text to parse.")

        text = response_text.strip()
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        logging.error(f"Could not extract valid JSON from response: {response_text[:200]}...")
        raise ValueError("Failed to extract valid JSON from LLM response.")

    # ------------------------------------------------------------------
    # Classification knowledge helpers (now no-ops)
    # ------------------------------------------------------------------

    def generate_classification_knowledge(self, product_description: str) -> str:
        logging.debug(
            "Classification knowledge generation is disabled (no external Gemini/GCV calls)."
        )
        return ""

    async def generate_classification_knowledge_async(self, product_description: str) -> str:
        logging.debug(
            "Async classification knowledge generation is disabled (no external Gemini/GCV calls)."
        )
        return ""

