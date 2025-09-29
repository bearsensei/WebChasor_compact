# WebChasor

A comprehensive AI-powered information processing system that combines intelligent routing, information retrieval, reasoning, and productivity tools. WebChasor provides structured, reliable responses through a hybrid approach that leverages multiple AI models and external data sources.

## Architecture

WebChasor uses a modular architecture with specialized action components:

- **Router**: Intelligent query classification using heuristics, lightweight classification, and LLM fallback
- **IR_RAG**: Information Retrieval and Retrieval-Augmented Generation for factual queries requiring external data
- **Reasoning**: Structured reasoning engine for analytical queries without external information needs
- **Productivity**: Text transformation tools for summarization, extraction, formatting, and analysis
- **Synthesizer**: Combines and synthesizes information from multiple sources into coherent responses
- **Config Manager**: Centralized configuration management with YAML-based settings

## Key Features

- **OpenAI API Compatible**: Full compatibility with OpenAI API format for easy integration
- **Streaming Support**: Real-time streaming responses with thinking process visualization
- **Multi-Model Support**: Configurable AI models for different components
- **External Data Integration**: SerpAPI and Google Search integration for real-time information
- **Structured Reasoning**: Scaffolded reasoning approaches for consistent, logical responses
- **Productivity Tools**: Built-in text processing and transformation capabilities

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd webchasor
```

2. Create and activate conda environment:
```bash
conda create -n webchaser python=3.9
conda activate webchaser
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
export OPENAI_API_KEY_AGENT="your-openai-api-key"
export SERPAPI_KEY="your-serpapi-key"  # Optional
```

5. Run the API server:
```bash
python src/api/api_server.py
```

## Configuration

The system uses `config/config.yaml` for comprehensive configuration. Key settings include:

- **Models**: Configure different AI models for each component (router, synthesizer, reasoning, etc.)
- **External Services**: API endpoints and rate limits for OpenAI, SerpAPI, Google Search
- **IR_RAG Settings**: Content processing, extraction, ranking, and search configuration
- **Productivity Tools**: Task-specific settings for summarization, translation, analysis
- **Performance**: Caching, concurrency, and timeout configurations

## API Usage

### OpenAI Compatible Endpoints

- `POST /v1/chat/completions` - Main chat completion endpoint
- `GET /v1/models` - List available models
- `GET /health` - Health check

### Example Usage

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "webchasor-thinking",
    "messages": [{"role": "user", "content": "What are the latest developments in AI?"}],
    "stream": true
  }'
```

## Components

### Action Types

- **IR_RAG**: Handles factual queries requiring external information retrieval
- **REASONING**: Manages analytical queries using structured reasoning scaffolds
- **PRODUCTIVITY**: Provides text transformation and analysis tools

### Query Categories

The router classifies queries into:
- INFORMATION_RETRIEVAL
- KNOWLEDGE_REASONING  
- TASK_PRODUCTIVITY
- CONVERSATIONAL_FOLLOWUP
- CREATIVE_GENERATION
- MULTIMODAL_QUERY

### Flow

See `flowchart.md` for the complete system flow diagram with detailed sequence diagrams.

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Quality
```bash
make ci  # Run all checks including linting and tests
```

## Security

- All API keys are stored in environment variables
- No sensitive information is committed to the repository
- Configurable rate limiting and timeout settings
- Safe search filtering for external queries

## License

MIT License
