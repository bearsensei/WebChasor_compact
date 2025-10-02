"""
SerpAPI Search Tool for WebChaser
Provides Google search results through SerpAPI
"""

import os
import json
import requests
from typing import List, Dict, Any, Union


class SerpAPISearch:
    """SerpAPI search tool for WebChaser"""
    
    name = "serpapi_search"
    description = "Search using SerpAPI (Google Search API alternative)"
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description': 'Search query string',
            'required': True
        },
        {
            'name': 'num_results',
            'type': 'integer',
            'description': 'Number of results to return (default: 10, max: 100)',
            'required': False
        },
        {
            'name': 'location',
            'type': 'string',
            'description': 'Location for localized search (e.g., "Hong Kong", "United States")',
            'required': False
        },
        {
            'name': 'language',
            'type': 'string',
            'description': 'Language code (e.g., "zh-cn", "en")',
            'required': False
        }
    ]

    def __init__(self):
        self.api_key = os.getenv('SERPAPI_KEY', '')
        if not self.api_key:
            print("Warning: SERPAPI_KEY environment variable not set")
        
        self.base_url = "https://serpapi.com/search"
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """Execute search and return formatted results"""
        try:
            # Parse parameters
            if isinstance(params, str):
                # Simple string query
                query = params
                num_results = kwargs.get('num_results', 10)
                location = kwargs.get('location', 'Hong Kong')
                language = kwargs.get('language', 'zh-cn')
                engine = kwargs.get('engine', 'google')
            else:
                # Dictionary parameters
                query = params.get('query', '')
                num_results = params.get('num_results', 10)
                location = params.get('location', 'Hong Kong')
                language = params.get('language', 'zh-cn')
                engine = params.get('engine', 'google')
            
            if not query:
                return "Error: Query parameter is required"
            
            # Execute search
            results = self._execute_search(query, num_results, location, language, engine)
            
            # Format results for display
            return self._format_results(results, query)
            
        except Exception as e:
            print(f"SerpAPI Search Error: {e}")
            return f"SerpAPI Search Error: {str(e)}"

    def get_structured_results(self, query: str, num_results: int = 10, location: str = 'Hong Kong', 
                             language: str = 'zh-cn', engine: str = 'google') -> List[Dict[str, Any]]:
        """Get structured search results for IR_RAG integration"""
        try:
            if not self.api_key:
                print("Error: SERPAPI_KEY not configured")
                return []
            
            return self._execute_search(query, num_results, location, language, engine)
        except Exception as e:
            print(f"SerpAPI Structured Search Error: {e}")
            return []

    def _execute_search(self, query: str, num_results: int, location: str, language: str, engine: str = 'google') -> List[Dict[str, Any]]:
        """Execute the actual SerpAPI search"""
        
        # Prepare search parameters
        search_params = {
            'engine': engine,  # 使用传入的 engine 参数
            'q': query,
            'api_key': self.api_key,
            'num': min(num_results, 10),  # SerpAPI max is 100
            'location': location,
            'hl': language,
            'gl': 'hk' if 'hong kong' in location.lower() else 'us',
            'safe': 'active'  # Enable safe search
        }
        
        print(f"SerpAPI Search: '{query}' (engine: {engine}, location: {location}, results: {num_results})")
        
        try:
            response = requests.get(self.base_url, params=search_params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'error' in data:
                print(f"SerpAPI Error: {data['error']}")
                return []
            
            # Extract results based on engine type
            organic_results = data.get('organic_results', [])
            news_results = data.get('news_results', [])
            
            # Also get answer box if available
            answer_box = data.get('answer_box', {})
            knowledge_graph = data.get('knowledge_graph', {})
            
            
            # Process results
            processed_results = []
            print(f"Raw API Response Keys: {list(data.keys())}")
            if 'answer_box' in data:
                print(f"Answer Box: {data['answer_box']}")
            if 'knowledge_graph' in data:
                print(f"Knowledge Graph: {data['knowledge_graph']}")
            # Add answer box first if available
            if answer_box:
                processed_results.append({
                    'type': 'answer_box',
                    'title': answer_box.get('title', ''),
                    'snippet': answer_box.get('snippet', ''),
                    'link': answer_box.get('link', ''),
                    'source': answer_box.get('displayed_link', '')
                })
            
            # Add knowledge graph if available
            if knowledge_graph:
                processed_results.append({
                    'type': 'knowledge_graph',
                    'title': knowledge_graph.get('title', ''),
                    'snippet': knowledge_graph.get('description', ''),
                    'link': knowledge_graph.get('website', ''),
                    'source': knowledge_graph.get('source', {}).get('name', '')
                })
            
            # Add news results (for google_news engine)
            if news_results and engine == 'google_news':
                for result in news_results[:num_results]:
                    processed_results.append({
                        'type': 'news',
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'link': result.get('link', ''),
                        'source': result.get('source', ''),
                        'date': result.get('date', ''),
                        'position': result.get('position', 0)
                    })
            
            # Add organic results (for regular google engine)
            if organic_results and engine != 'google_news':
                for result in organic_results[:num_results]:
                    processed_results.append({
                        'type': 'organic',
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'link': result.get('link', ''),
                        'source': result.get('displayed_link', ''),
                        'position': result.get('position', 0)
                    })
            
            print(f"SerpAPI returned {len(processed_results)} results")
            return processed_results
            
        except requests.RequestException as e:
            print(f"SerpAPI Request Error: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"SerpAPI JSON Error: {e}")
            return []

    def _format_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results for display"""
        
        if not results:
            return f"No results found for: {query}"
        
        formatted = f"**SerpAPI Search Results for:** {query}\n\n"
        
        for i, result in enumerate(results, 1):
            result_type = result.get('type', 'organic')
            title = result.get('title', 'No title')
            snippet = result.get('snippet', 'No description available')
            link = result.get('link', '')
            source = result.get('source', '')
            
            # Add type indicator
            if result_type == 'answer_box':
                formatted += "**Answer Box:**\n"
            elif result_type == 'knowledge_graph':
                formatted += "**Knowledge Graph:**\n"
            elif result_type == 'news':
                formatted += f"{i}. "
            else:
                formatted += f"{i}. "
            
            # Add title
            if link:
                formatted += f"{title}\n"
            else:
                formatted += f"{title}\n"
            
            # Add snippet
            if snippet:
                # Clean and truncate snippet
                clean_snippet = snippet.replace('\n', ' ').strip()
                if len(clean_snippet) > 200:
                    clean_snippet = clean_snippet[:200] + "..."
                formatted += f"{clean_snippet}\n"
            
            # Add date for news results
            if result_type == 'news' and result.get('date'):
                formatted += f"Date: {result.get('date')}\n"
            
            # Add source and link
            if source and link:
                formatted += f"Source: {source}\n"
                formatted += f"Link: {link}\n"
            elif link:
                formatted += f"Link: {link}\n"
            
            formatted += "\n"
        
        return formatted

# Test function
def test_serpapi():
    """Test SerpAPI functionality"""
    search_tool = SerpAPISearch()
    
    # Test query
    test_query = "香港天气"
    result = search_tool.call({'query': test_query, 'num_results': 5})
    
    print("=" * 60)
    print("SerpAPI Test Results:")
    print("=" * 60)
    print(result)

if __name__ == "__main__":
    test_serpapi() 