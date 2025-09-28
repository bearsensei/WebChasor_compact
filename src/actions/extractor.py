# Chasor is a system that can extract, calculate, synthesize, retrieve, and reorganize information. Extractor is a class that can extract information from a text. Calculator is a class that can calculate information. Synthesizer is a class that can synthesize information. InfoRetriever is a class that can retrieve information from the search engine or a database. InfoReorganizer is a class that can reorganize the retrieved information into a structured format.
import os
from dotenv import load_dotenv
from serpapi_search import SerpAPISearch
load_dotenv()

GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


class Extractor:
# the task extractor got is like: {"plan": {"tasks_to_extract": [{"fact": "What year was Marie Curie born?", "variable_name": "marie_birth_year"}, {"fact": "What year was Pierre Curie born?", "variable_name": "pierre_birth_year"}], "final_calculation": {"operation": "subtract", "operands": ["marie_birth_year", "pierre_birth_year"]}}}

    def extract(self, variable_name, fact, GOOGLE_SEARCH_KEY, GOOGLE_CSE_ID):
        """Extract information using SerpAPI search"""
        try:
            # Initialize SerpAPI search tool
            search_tool = SerpAPISearch()
            
            # Prepare search parameters
            search_params = {
                'query': fact,
                'num_results': 5,  # Get top 5 results for better accuracy
                'location': 'Hong Kong',  # Default location
                'language': 'zh-cn'  # Default language
            }
            
            print(f"🔍 Extracting: {fact} (variable: {variable_name})")
            
            # Execute search using SerpAPI
            search_results = search_tool.call(search_params)
            
            if search_results:
                return search_results
            else:
                print(f"⚠️ No results found for: {fact}")
                return None

            
            
        except Exception as e:
            print(f"Extraction error for {variable_name}: {str(e)}")
            return None

    def _extract_relevant_info(self, search_results, fact, variable_name):
        """Extract the most relevant information from search results"""
        try:
            # Parse the search results to find the most relevant answer
            # This is a simplified extraction - you might want to enhance this
            
            # Look for patterns that might contain the answer
            # For example, if looking for a year, look for 4-digit numbers
            if "year" in fact.lower() or "born" in fact.lower():
                # Extract year information
                import re
                year_pattern = r'\b(19|20)\d{2}\b'
                years = re.findall(year_pattern, search_results)
                if years:
                    structured_info = {
                        "variable_name": variable_name,
                        "fact": fact,
                        "year": years[0]
                    }
                    return structured_info  # Return the first year found
            
            # For other types of facts, you might want to use more sophisticated NLP
            # For now, return the first snippet as a fallback
            lines = search_results.split('\n')
            for line in lines:
                if line.strip() and not line.startswith('🔍') and not line.startswith('**'):
                    # Clean up the line and return it
                    clean_line = line.replace('**', '').replace('��', '').replace('📚', '').strip()
                    if len(clean_line) > 10:  # Only return substantial content
                        return clean_line[:200]  # Limit length
            
            return "Information not found"
            
        except Exception as e:
            print(f"❌ Error extracting relevant info: {str(e)}")
            return None

if __name__ == "__main__":
    extractor = Extractor()
    test_result = extractor.extract("marie_birth_year", "What year was Marie Curie born?", GOOGLE_SEARCH_KEY, GOOGLE_CSE_ID)
    print(test_result)