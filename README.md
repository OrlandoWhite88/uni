# HTS Code Classification System

A sophisticated AI-powered system for automated HTS (Harmonized Tariff Schedule) classification using advanced multi-hypothesis beam search algorithms and interactive clarification.

## 🏗️ Project Architecture

This project has been refactored into a **modular architecture** for better maintainability and separation of concerns:

### Core Modules (`api/` directory)

```
api/
├── models.py                    # Data structures (HTSNode, ClassificationPath, etc.)
├── tree_navigator.py            # Tree operations & navigation logic
├── llm_client.py               # LLM communication (Groq, Vertex AI)
├── classification_engine.py    # Core algorithms + configuration + question generation
├── streaming_engine.py         # Server-Sent Events streaming
├── groq_tree_engine.py         # Main orchestrator (thin facade)
├── main.py                     # FastAPI web server
├── serialization_utils.py      # State serialization/deserialization
└── hts_data.json              # HTS tariff data
```

### Key Features

- **Multi-Hypothesis Beam Search**: Explores multiple classification paths simultaneously
- **Interactive Clarification**: Asks targeted questions when confidence is low
- **Real-time Streaming**: Server-Sent Events for live classification updates
- **Multiple Engines**: Support for Groq, Cerebras, and traditional tree engines
- **CLI Interface**: Command-line tool for local testing
- **Web API**: RESTful endpoints for integration

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Required API Keys**:
   - `GROQ_API_KEY` - For Groq LLM engine
   - `GOOGLE_APPLICATION_CREDENTIALS_JSON_B64` - For Vertex AI (optional)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd hscode

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY="your-groq-api-key"
export GOOGLE_APPLICATION_CREDENTIALS_JSON_B64="your-base64-encoded-service-account"
```

### Running the System

#### 1. CLI Interface (Recommended for Testing)

```bash
# Interactive classification with questions
python cli_classifier.py --product "Bolt Action Sniper Rifle Single Shot" --interactive --engine groq

# Single-shot classification
python cli_classifier.py --product "leather shoes" --engine groq

# Compare different modes
python cli_classifier.py --product "steel kitchen knife" --compare --engine groq

# Batch classification
python cli_classifier.py --batch products.txt --engine groq

# Interactive menu
python cli_classifier.py
```

#### 2. Web API Server

```bash
# Start the FastAPI server
uvicorn api.main:app --reload --port 8000

# Access the API documentation
open http://localhost:8000/docs
```

#### 3. Direct Script Usage

```bash
# Run direct classification
python cli_classifier_direct.py "laptop computer"
```

## 🔧 Configuration

### Engine Selection

The system supports multiple classification engines:

- **`groq`** (default): Uses Groq's LLM with advanced beam search
- **`cerebras`**: Alternative LLM engine  
- **`tree`**: Traditional tree-based classification

### Configuration Constants

Key parameters can be adjusted in `api/classification_engine.py`:

```python
# Beam search configuration
CHAPTER_BEAM_SIZE = 3      # Initial chapter hypotheses
CLASSIFICATION_BEAM_SIZE = 6  # Beam size for detailed classification

# Confidence threshold for questions
CONFIDENCE_THRESHOLD = 0.85   # When to ask clarification questions

# LLM settings
LLM_TEMPERATURE = 0          # Deterministic vs creative responses
```

### Environment Variables

```bash
# Required
GROQ_API_KEY=your-groq-api-key

# Optional
GOOGLE_APPLICATION_CREDENTIALS_JSON_B64=base64-encoded-service-account
LOG_PROMPTS=true                    # Enable prompt logging
PROMPT_LOG_FILE=groq_prompts.log    # Custom prompt log file
PATH_WORKERS=10                     # Concurrent path processing
CALIBRATE_WORKERS=10                # Concurrent candidate scoring
```

#### Custom vLLM / OpenAI-Compatible Backend

The `LLMClient` can route all Responses API traffic to your own OpenAI-compatible server (for example, a self-hosted vLLM node). Set the following variables to make it the default backend for the tree engine and CLI:

```bash
# Point the OpenAI SDK at your server (must include /v1)
CUSTOM_OPENAI_BASE_URL="http://your-host:8000/v1"

# Optional overrides
CUSTOM_OPENAI_API_KEY="sk-local"          # Only if your server enforces auth
CUSTOM_OPENAI_MODEL="openai/gpt-oss-120b" # Defaults to this model name
CUSTOM_OPENAI_MAX_OUTPUT_TOKENS=8192      # Cap for responses.create
CUSTOM_OPENAI_TIMEOUT=90                  # Seconds before timing out
CUSTOM_OPENAI_REASONING_EFFORT=medium     # low | medium | high
```

When `CUSTOM_OPENAI_BASE_URL` is present the system automatically:

- Uses the unified system prompt from `api/unified_system_prompt.py` as the `system` message for every call.
- Sends the user prompt exactly as produced by the tree engine.
- Falls back to OpenAI's `gpt-5` Responses API only if the custom server is not configured or `OPENAI_PROVIDER=openai` is set.

Additional toggles:

- `OPENAI_PROVIDER=openai` forces the original OpenAI platform even when a custom base URL is configured.
- `OPENAI_RESPONSES_MODEL=gpt-5` lets you change the upstream OpenAI model name.
- `OPENAI_REASONING_EFFORT=low` controls the Responses API reasoning config.
- `OPENAI_STORE_RESPONSES=true` keeps compatibility with OpenAI's `store` flag (disabled automatically for custom servers).

> **Tip:** The CLI default engine (`groq`) already uses `LLMClient`, so once the environment variables above are set you can immediately test your hosted model with `python cli_classifier.py --product "..."`

## 📋 API Endpoints

### Classification Endpoints

#### Start Classification
```http
POST /classify
Content-Type: application/json

{
  "product": "steel kitchen knife",
  "interactive": true,
  "max_questions": 3,
  "use_multi_hypothesis": true,
  "hypothesis_count": 3
}
```

#### Continue Classification
```http
POST /classify/continue
Content-Type: application/json

{
  "state": {...},
  "answer": "It has a sharp cutting edge",
  "interactive": true,
  "max_questions": 3
}
```

#### Streaming Classification
```http
GET /classify/stream?product=laptop&interactive=true
Accept: text/event-stream
```

### Tariff Information Endpoints

#### Get Tariff Details
```http
GET /tariff_details/8471.30.01
```

#### Explain Tariff
```http
GET /explain_tariff/8471.30.01
```

#### Get Tree Subtree
```http
GET /subtree/8471
```

## 🧪 Testing & Development

### Running Tests

```bash
# Test CLI classification
python cli_classifier.py --product "test product" --debug

# Test streaming
open test_streaming_complete.html

# Test batch processing
echo -e "laptop\nshoes\nknife" > test_products.txt
python cli_classifier.py --batch test_products.txt
```

### Debug Mode

```bash
# Enable detailed logging
python cli_classifier.py --debug --product "laptop"

# View logs
tail -f classifier_cli.log
```

### Development Workflow

1. **Modify Core Logic**: Edit `api/classification_engine.py`
2. **Change Tree Operations**: Edit `api/tree_navigator.py`
3. **Update LLM Communication**: Edit `api/llm_client.py`
4. **Add New Endpoints**: Edit `api/main.py`
5. **Test Changes**: Use CLI tool for quick testing

## 📁 File Organization

### Main Components

- **`api/groq_tree_engine.py`**: Main orchestrator (400 lines, down from 3000+)
- **`api/classification_engine.py`**: Core classification algorithms and question generation
- **`api/tree_navigator.py`**: HTS tree building and navigation
- **`api/llm_client.py`**: Groq and Vertex AI communication
- **`api/streaming_engine.py`**: Real-time event streaming
- **`api/models.py`**: Data structures and utilities

### Support Files

- **`cli_classifier.py`**: Command-line interface
- **`cli_classifier_direct.py`**: Direct classification script
- **`requirements.txt`**: Python dependencies
- **`vercel.json`**: Vercel deployment configuration
- **`api/hts_data.json`**: HTS classification tree data
- **`tarrif.xlsx`**: Tariff rate data

### Test Files

- **`test_streaming_*.html`**: Streaming interface tests
- **`STREAMING_TEST_GUIDE.md`**: Streaming test documentation
- **`CLI_README.md`**: CLI-specific documentation

## 🔄 Classification Process

### Multi-Hypothesis Beam Search

1. **Chapter Selection**: Identify top 3 most likely chapters
2. **Path Initialization**: Create classification paths for each chapter
3. **Beam Advancement**: Score candidates and maintain top K paths
4. **Confidence Checking**: Ask questions when confidence < 0.85
5. **Termination**: Return best path when complete or confident

### Interactive Clarification

When the system's confidence is below the threshold, it generates targeted questions:

```
❓ Does your steel kitchen knife have a serrated cutting edge 
   (for cutting through tough materials like bread or meat), 
   or a smooth straight edge (for precise slicing)?

1. Serrated edge - designed for sawing action through tough materials
2. Smooth straight edge - designed for clean, precise cuts
```

### Streaming Updates

Real-time events provide insight into the classification process:

- `classification_start`: Begin classification
- `chapter_selection`: Top chapters identified  
- `beam_advancement`: Paths being evaluated
- `question_generated`: Clarification needed
- `classification_complete`: Final result

## 🚀 Deployment

### Local Development

```bash
uvicorn api.main:app --reload --port 8000
```

### Vercel Deployment

The project is configured for Vercel deployment with `vercel.json`. Set environment variables in Vercel dashboard:

- `GROQ_API_KEY`
- `GOOGLE_APPLICATION_CREDENTIALS_JSON_B64`

### Docker Deployment

```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📊 Performance & Monitoring

### Logging

The system provides comprehensive logging:

- **Console Logging**: Real-time feedback during classification
- **File Logging**: Detailed logs in `classifier_cli.log`
- **Prompt Logging**: LLM interactions (when `LOG_PROMPTS=true`)

### Metrics

Key performance indicators:

- **Classification Accuracy**: Multi-hypothesis vs single-path comparison
- **Question Efficiency**: Average questions needed per classification
- **Response Time**: Time to classification completion
- **Confidence Levels**: Distribution of final confidence scores

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from project root
2. **API Key Missing**: Set `GROQ_API_KEY` environment variable
3. **JSON Parsing**: LLM responses may need retry logic
4. **Memory Usage**: Large beam sizes can consume significant memory

### Debug Commands

```bash
# Test engine import
python -c "from api.groq_tree_engine import HTSTree; print('OK')"

# Test API connection
python -c "from api.llm_client import LLMClient; client = LLMClient(); print('Connected')"

# Validate HTS data
python -c "import json; data = json.load(open('api/hts_data.json')); print(f'{len(data)} items loaded')"
```

## 🤝 Contributing

### Code Organization Guidelines

- **Single Responsibility**: Each module has one clear purpose
- **Dependency Injection**: Engines are composed, not inherited
- **Interface Consistency**: Maintain backward compatibility
- **Error Handling**: Graceful degradation and informative errors

### Adding New Features

1. **New Classification Algorithms**: Add to `classification_engine.py`
2. **New LLM Providers**: Extend `llm_client.py`
3. **New Tree Operations**: Add to `tree_navigator.py`
4. **New API Endpoints**: Add to `main.py`

## 📜 License

[Specify your license here]

## 📞 Support

For issues and questions:
- Check the logs in `classifier_cli.log`
- Use debug mode: `--debug`
- Review the API documentation at `/docs`
- Test with the CLI interface first
