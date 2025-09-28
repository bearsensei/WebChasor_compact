# WebChasor Compact

A compact implementation of the WebChasor system - an AI-powered information extraction, calculation, synthesis, retrieval, and reorganization system.

## Architecture

WebChasor uses a hybrid routing approach with multiple processing components:

- **Router**: Classifies queries using fast heuristics, lightweight classification, and LLM fallback
- **Planner**: Creates execution plans for complex information retrieval tasks  
- **Extractor**: Extracts specific information from text and search results
- **Calculator**: Performs mathematical calculations
- **Synthesizer**: Combines and synthesizes information from multiple sources
- **InfoRetriever**: Retrieves information from search engines and databases
- **InfoReorganizer**: Reorganizes retrieved information into structured formats

## Setup

1. Clone the repository:
```bash
git clone https://github.com/bearsensei/WebChasor_compact.git
cd WebChasor_compact
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run the system:
```bash
python src/chasor.py
```

## Configuration

Copy `.env.example` to `.env` and configure the following:

- `OPENAI_API_KEY_AGENT`: Your OpenAI API key
- `SERPAPI_KEY`: Your SerpAPI key (optional)
- `GOOGLE_SEARCH_KEY`: Google Custom Search API key (optional)

## Components

### Router
The router classifies user queries into categories:
- INFORMATION_RETRIEVAL
- MATH_QUERY  
- TASK_PRODUCTIVITY
- KNOWLEDGE_REASONING
- CONVERSATIONAL_FOLLOWUP
- CREATIVE_GENERATION
- MULTIMODAL_QUERY

### Flow
See `flowchart.md` for the complete system flow diagram.

## Security

- All API keys are stored in environment variables
- No sensitive information is committed to the repository
- Use `.env.example` as a template for configuration

## License

MIT License
