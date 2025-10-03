# Search Provider Configuration Guide

WebChasor supports multiple search API providers for information retrieval. You can choose between SerpAPI (third-party), Google Custom Search API (official Google), or GCP Vertex AI Search (Google Cloud Discovery Engine).

## Available Providers

### 1. SerpAPI (Default)
- **Provider**: Third-party service wrapping Google Search
- **Pros**: 
  - Easy to set up
  - Rich structured data (knowledge graphs, answer boxes, etc.)
  - Supports multiple search engines (Google, Bing, Yahoo, etc.)
  - Better rate limits
- **Cons**: 
  - Requires paid subscription
  - Additional cost per search

### 2. Google Custom Search API
- **Provider**: Official Google API
- **Pros**: 
  - Official Google service
  - Free tier available (100 searches/day)
  - Direct integration with Google
- **Cons**: 
  - Limited to 10 results per request
  - Free tier has strict quota limits
  - Requires Custom Search Engine setup

### 3. GCP Vertex AI Search (NEW)
- **Provider**: Google Cloud Discovery Engine
- **Pros**:
  - Enterprise-grade search powered by Google's AI
  - Customizable search index
  - Advanced relevance tuning
  - Support for custom data sources
  - Good for domain-specific search
- **Cons**:
  - Requires Google Cloud Platform account
  - More complex setup (create search engine/data store)
  - Pricing based on query volume and data storage

## Configuration

### Step 1: Choose Your Provider

Edit `config/config.yaml`:

```yaml
ir_rag:
  search:
    provider: serpapi  # Options: 'serpapi', 'google_custom_search', or 'gcp_vertex_search'
```

### Step 2: Set Environment Variables

Create a `.env` file in the project root:

#### For SerpAPI:
```bash
SERPAPI_KEY=your-serpapi-key-here
```

#### For Google Custom Search API:
```bash
GOOGLE_SEARCH_KEY=your-google-api-key-here
GOOGLE_CSE_ID=your-custom-search-engine-id-here
```

#### For GCP Vertex AI Search:
```bash
GCP_PROJECT_ID=your-gcp-project-id
GCP_ENGINE_ID=your-vertex-search-engine-id
GCP_API_KEY=your-gcp-api-key
GCP_LOCATION=global  # Optional, default: global
```

### Step 3: Obtain API Credentials

#### SerpAPI Setup:
1. Visit [https://serpapi.com/](https://serpapi.com/)
2. Sign up for an account
3. Go to Dashboard → API Key
4. Copy your API key to `.env`

#### Google Custom Search API Setup:
1. **Get API Key**:
   - Visit [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing
   - Enable "Custom Search API"
   - Go to "Credentials" → "Create Credentials" → "API Key"
   - Copy the API key to `.env`

2. **Create Custom Search Engine**:
   - Visit [Programmable Search Engine](https://programmablesearchengine.google.com/)
   - Click "Add" to create a new search engine
   - **Sites to search**: Enter `www.google.com` (to search the entire web)
   - **Name**: WebChasor Search (or any name you prefer)
   - Click "Create"
   - Go to "Setup" → "Basics"
   - Copy the "Search engine ID" to `.env` as `GOOGLE_CSE_ID`
   - **Important**: Turn ON "Search the entire web" option

#### GCP Vertex AI Search Setup:
1. **Create GCP Project**:
   - Visit [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing
   - Enable billing for the project

2. **Enable Discovery Engine API**:
   - Go to "APIs & Services" → "Library"
   - Search for "Discovery Engine API"
   - Click "Enable"

3. **Create Search Engine**:
   - Go to "Discovery Engine" in the console
   - Click "Create App" → "Search"
   - Choose "Generic" or "Website" based on your needs
   - Configure your data source (web URLs, documents, etc.)
   - Note the Engine ID from the app details

4. **Get API Key**:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "API Key"
   - Copy the API key to `.env` as `GCP_API_KEY`

5. **Set Project ID**:
   - Your project ID is visible at the top of the Cloud Console
   - Copy it to `.env` as `GCP_PROJECT_ID`

## Feature Comparison

| Feature | SerpAPI | Google Custom Search | GCP Vertex AI Search |
|---------|---------|---------------------|---------------------|
| Results per request | Up to 100 | Max 10 | Configurable |
| Knowledge Graph | ✓ | ✗ | ✗ |
| Answer Box | ✓ | ✗ | ✗ |
| News Results | ✓ | Limited | Custom |
| Rich Metadata | ✓ | Basic | Extensive |
| Time Filtering | ✓ (tbs param) | Limited | Custom |
| Custom Data Sources | ✗ | ✗ | ✓ |
| Relevance Tuning | ✗ | ✗ | ✓ |
| Free Tier | 100 searches | 100 searches/day | Limited trial |
| Cost (paid) | $50/5K searches | $5/1K searches | Varies by usage |

## Usage Examples

### Basic Search

```python
from actions.serpapi_search import SerpAPISearch
from actions.google_search import GoogleCustomSearch
from actions.gcp_vertex_search import GCPVertexSearch

# Using SerpAPI
serpapi = SerpAPISearch()
results = serpapi.get_structured_results("香港天气", num_results=10)

# Using Google Custom Search
google_search = GoogleCustomSearch()
results = google_search.get_structured_results("香港天气", num_results=10)

# Using GCP Vertex AI Search
gcp_search = GCPVertexSearch()
results = gcp_search.get_structured_results("香港天气", num_results=10)
```

### Time-Filtered Search

```python
# Search for results from the past month
results = serpapi.get_structured_results(
    "浸会大学校长", 
    num_results=10,
    tbs="qdr:m"  # Past month
)

# Time filter options:
# qdr:h  - Past hour
# qdr:d  - Past 24 hours
# qdr:w  - Past week
# qdr:m  - Past month
# qdr:y  - Past year
```

### Batch Search

```python
queries = ["香港天气", "香港股市", "香港新闻"]

# Batch search with concurrent execution
results_map = serpapi.batch_call(
    queries,
    num_results=5,
    location="Hong Kong",
    language="zh-cn"
)

# Flatten results
flat_results = serpapi.batch_call_flat(queries, num_results=5)
```

### In IR_RAG Action

The search provider is automatically selected based on `config.yaml`:

```python
from actions.ir_rag import IR_RAG

# Will use the provider specified in config.yaml
ir_rag = IR_RAG(llm_client=client)
result = await ir_rag.execute(context)
```

## Testing

Run the test script to verify both providers:

```bash
# Ensure environment variables are set
export SERPAPI_KEY=your-key
export GOOGLE_SEARCH_KEY=your-key
export GOOGLE_CSE_ID=your-cse-id
export GCP_PROJECT_ID=your-project-id
export GCP_ENGINE_ID=your-engine-id
export GCP_API_KEY=your-api-key

# Run test
python test_search_providers.py
```

## Switching Providers

To switch between providers at runtime:

1. **Via config.yaml** (Recommended):
   ```yaml
   ir_rag:
     search:
       provider: serpapi  # Options: 'serpapi', 'google_custom_search', 'gcp_vertex_search'
   ```

2. **Via Code** (Advanced):
   ```python
   from actions.ir_rag import IR_RAG, IRConfig, RetrievalProvider
   
   # Use Google Custom Search
   config = IRConfig(
       search_provider=RetrievalProvider.GOOGLE_CUSTOM_SEARCH
   )
   
   # Or use GCP Vertex AI Search
   config = IRConfig(
       search_provider=RetrievalProvider.GCP_VERTEX_SEARCH
   )
   
   ir_rag = IR_RAG(config=config, llm_client=client)
   ```

## Troubleshooting

### SerpAPI Issues
- **Error: Invalid API Key**: Check that `SERPAPI_KEY` is correctly set
- **Error: Rate Limit**: Adjust `qps` in `config.yaml` or upgrade plan
- **Empty Results**: Check query syntax and location settings

### Google Custom Search Issues
- **Error: API Key Invalid**: Verify API key and ensure Custom Search API is enabled
- **Error: Invalid CSE ID**: Check `GOOGLE_CSE_ID` is correct
- **No Results**: Ensure "Search the entire web" is enabled in CSE settings
- **Quota Exceeded**: Free tier limited to 100 searches/day

### GCP Vertex AI Search Issues
- **Error: Missing configuration**: Ensure `GCP_PROJECT_ID`, `GCP_ENGINE_ID`, and `GCP_API_KEY` are all set
- **Error: API not enabled**: Enable "Discovery Engine API" in Google Cloud Console
- **Error: Permission denied**: Verify API key has proper permissions for Discovery Engine
- **Error: Engine not found**: Check that your search engine is created and the ID is correct
- **Error: RESOURCE_EXHAUSTED**: Rate limit exceeded, adjust `qps` in config or increase quota
- **Library not installed**: Run `pip install google-cloud-discoveryengine`

### Common Issues
- **Import Error**: Ensure all dependencies installed: `pip install -r requirements.txt`
- **No Results**: Check internet connection and API status
- **Timeout**: Increase timeout in `config.yaml`: `ir_rag.web_scraping.timeout`

## Rate Limits and Costs

### SerpAPI
- Free: 100 searches/month
- Developer: $50/month (5,000 searches)
- Production: $125/month (15,000 searches)

### Google Custom Search API
- Free: 100 searches/day (36,500/year)
- Paid: $5 per 1,000 queries (after free tier)

### GCP Vertex AI Search
- Free: Limited trial credits (varies by GCP account)
- Paid: Based on query volume and data storage
  - ~$0.01 per search query
  - Additional costs for data storage and indexing

## Recommendations

**Use SerpAPI if**:
- You need rich structured data (knowledge graphs, etc.)
- You want more than 10 results per query
- Budget allows ($50+/month)
- You need better rate limits

**Use Google Custom Search if**:
- You're on a tight budget (100 free searches/day)
- 10 results per query is sufficient
- You prefer official Google API
- Low-to-medium query volume

**Use GCP Vertex AI Search if**:
- You need enterprise-grade search with AI capabilities
- You want to index your own custom data sources
- You need advanced relevance tuning and personalization
- You have a GCP infrastructure already
- Budget allows for per-query pricing

## Advanced Configuration

### Customize Search Parameters

In `config/config.yaml`:

```yaml
ir_rag:
  search:
    provider: serpapi              # Choose provider
    location: Hong Kong            # Search location
    language: zh-cn               # Search language
    max_results: 20               # Max results to retrieve
    concurrent: 8                 # Concurrent searches
    qps: 5                        # Queries per second
    retries: 2                    # Retry attempts
```

### Multiple Provider Support (Future)

WebChasor is designed to support fallback mechanisms:

```python
# Future: Automatic fallback
# If SerpAPI fails, automatically try Google Custom Search
config = IRConfig(
    search_provider=RetrievalProvider.HYBRID,
    fallback_providers=[
        RetrievalProvider.SERPAPI,
        RetrievalProvider.GOOGLE_CUSTOM_SEARCH
    ]
)
```

---

**Last Updated**: 2025-10-03

