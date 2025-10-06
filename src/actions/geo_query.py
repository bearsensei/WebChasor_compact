"""
Geographic Query Action for WebChasor
Handles location-based queries using IntegratedGeoTool
"""
import re
import asyncio
from typing import Optional
from dataclasses import dataclass

from artifacts import Action, Context, Artifact
from actions.integrated_geo_tool import IntegratedGeoTool
from config_manager import get_config


@dataclass
class GeoIntent:
    """Parsed geographic query intent"""
    intent_type: str  # "nearby_search", "route", "distance"
    center_location: Optional[str] = None
    target_place: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    radius: int = 2000  # Default 2km
    travel_mode: str = "TRANSIT"  # TRANSIT, DRIVE, WALK


class GEO_QUERY(Action):
    """Action for handling geographic/location queries"""
    
    name = "GEO_QUERY"
    requires_tools = []
    max_time_s = 30
    max_tokens_in = 2000
    max_tokens_out = 3000
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Google Maps API key"""
        try:
            self.geo_tool = IntegratedGeoTool(api_key=api_key)
            print(f"[GEO_QUERY][INIT] IntegratedGeoTool initialized successfully")
        except ValueError as e:
            print(f"[GEO_QUERY][WARN] {e}")
            print(f"[GEO_QUERY][WARN] Geographic queries will return error messages")
            self.geo_tool = None
        
        # Load config for enhancement
        cfg = get_config()
        self.enable_enhancement = cfg.get('geo_query.enable_enhancement', True)
        self.enhancement_temperature = cfg.get('geo_query.enhancement_temperature', 0.7)
        
        # Initialize synthesizer if enhancement is enabled
        self.synthesizer = None
        if self.enable_enhancement:
            try:
                from synthesizer import Synthesizer
                self.synthesizer = Synthesizer()
                print(f"[GEO_QUERY][INIT] Synthesizer initialized for enhancement")
            except Exception as e:
                print(f"[GEO_QUERY][WARN] Failed to initialize synthesizer: {e}")
                self.enable_enhancement = False
    
    async def run(self, ctx: Context, toolset) -> Artifact:
        """Execute geographic query"""
        print(f"[GEO_QUERY][START] query='{ctx.query}'")
        
        if not self.geo_tool:
            return Artifact(
                kind="text",
                content="Geographic query service is temporarily unavailable. Please configure GOOGLE_MAP_KEY environment variable.",
                meta={"error": "geo_tool_unavailable"}
            )
        
        try:
            # Parse query intent
            intent = self._parse_geo_intent(ctx.query)
            print(f"[GEO_QUERY][INTENT] type={intent.intent_type}, center={intent.center_location}, target={intent.target_place}")
            
            # Execute based on intent type
            if intent.intent_type == "nearby_search":
                result = await self._handle_nearby_search(intent)
            elif intent.intent_type == "route":
                result = await self._handle_route_query(intent)
            elif intent.intent_type == "distance":
                result = await self._handle_distance_query(intent)
            else:
                result = self._fallback_response(ctx.query)
            
            # Enhance response if enabled
            if self.enable_enhancement and self.synthesizer:
                result = await self._enhance_response(result, ctx)
            
            print(f"[GEO_QUERY][COMPLETE]")
            return result
            
        except Exception as e:
            print(f"[GEO_QUERY][ERROR] {e}")
            import traceback
            traceback.print_exc()
            return Artifact(
                kind="text",
                content=f"Error processing geographic query: {str(e)}",
                meta={"error": str(e)}
            )
    
    def _parse_geo_intent(self, query: str) -> GeoIntent:
        """Parse query to determine geographic intent"""
        query_lower = query.lower()
        
        # Detect nearby search patterns (Chinese)
        nearby_patterns = [
            r'(附近|周边|附近有|周边有)(什么|哪里有|有没有)(.+)',
            r'(.+)(附近|周边)有(什么|哪些)(.+)',
            r'(找|搜索|查找)(.+)(附近|周边)的(.+)',
            r'(.+)(附近|周边)的(.+)',
        ]
        
        for pattern in nearby_patterns:
            match = re.search(pattern, query)
            if match:
                center = self._extract_location(query)
                target = self._extract_target(query)
                print(f"[GEO_QUERY][PARSE] Nearby search detected: center='{center}', target='{target}'")
                return GeoIntent(
                    intent_type="nearby_search",
                    center_location=center,
                    target_place=target,
                    radius=3000
                )
        
        # Detect route patterns (Chinese) - strict patterns first
        strict_route_patterns = [
            r'(从|由)(.+)(到|去|往)(.+)(怎么走|路线|交通|怎么去)',
            r'(.+)(到|去|往)(.+)(的路线|怎么走|交通|怎么去)',
        ]
        
        for pattern in strict_route_patterns:
            match = re.search(pattern, query)
            if match:
                groups = match.groups()
                origin = groups[1].strip() if len(groups) > 1 else None
                destination = groups[3].strip() if len(groups) > 3 else None
                print(f"[GEO_QUERY][PARSE] Route query detected (strict): origin='{origin}', destination='{destination}'")
                return GeoIntent(
                    intent_type="route",
                    origin=origin,
                    destination=destination,
                    travel_mode="TRANSIT"
                )
        
        # Lenient route pattern: "从A到B" (only if it looks like place names)
        # Must start with "从" and contain "到", and not contain nearby/周边 keywords
        if re.match(r'(从|由).+(到|去|往).+', query) and not re.search(r'(附近|周边)', query):
            match = re.search(r'(从|由)(.+?)(到|去|往)(.+?)(?:\?|？|$)', query)
            if match:
                groups = match.groups()
                origin = groups[1].strip() if len(groups) > 1 else None
                destination = groups[3].strip() if len(groups) > 3 else None
                # Additional check: origin and destination should be short (likely place names)
                if origin and destination and len(origin) <= 20 and len(destination) <= 20:
                    print(f"[GEO_QUERY][PARSE] Route query detected (lenient): origin='{origin}', destination='{destination}'")
                    return GeoIntent(
                        intent_type="route",
                        origin=origin,
                        destination=destination,
                        travel_mode="TRANSIT"
                    )
        
        # Default: treat as nearby search with current location
        print(f"[GEO_QUERY][PARSE] Defaulting to nearby search")
        return GeoIntent(
            intent_type="nearby_search",
            center_location="当前位置",
            target_place=query,
            radius=2000
        )
    
    def _extract_location(self, query: str) -> str:
        """Extract center location from query"""
        # Look for location keywords
        location_keywords = ['附近', '周边']
        for keyword in location_keywords:
            if keyword in query:
                # Extract text before the keyword
                parts = query.split(keyword)[0].strip()
                if parts and len(parts) > 1:
                    return parts
        
        # If no clear location, use default
        return "当前位置"
    
    def _extract_target(self, query: str) -> str:
        """Extract target place type from query"""
        # Common place types mapping
        place_mapping = {
            '咖啡厅': 'cafe',
            '咖啡店': 'cafe',
            '餐厅': 'restaurant',
            '饭店': 'restaurant',
            '超市': 'supermarket',
            '药店': 'pharmacy',
            '医院': 'hospital',
            '银行': 'bank',
            'ATM': 'ATM',
            '加油站': 'gas station',
            '便利店': 'convenience store',
            '书店': 'bookstore',
            '电影院': 'cinema',
            '健身房': 'gym',
            'KFC': 'KFC',
            '肯德基': 'KFC',
            '麦当劳': 'McDonalds',
            '星巴克': 'Starbucks',
        }
        
        for keyword, place_type in place_mapping.items():
            if keyword.lower() in query.lower():
                return place_type
        
        # Extract target from query (remove location keywords)
        target = query
        for keyword in ['附近', '周边', '有什么', '哪里有', '有没有', '的', '找', '搜索', '查找']:
            target = target.replace(keyword, '')
        
        return target.strip() or query
    
    async def _handle_nearby_search(self, intent: GeoIntent) -> Artifact:
        """Handle nearby place search"""
        print(f"[GEO_QUERY][NEARBY] Searching for '{intent.target_place}' near '{intent.center_location}' within {intent.radius}m")
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.geo_tool.find_places_simple,
                intent.center_location,
                intent.target_place,
                intent.radius,
                intent.travel_mode,
                5
            )
            
            if not result or result.get('total_found', 0) == 0:
                return Artifact(
                    kind="text",
                    content=f"Sorry, no {intent.target_place} found near {intent.center_location}.",
                    meta={"search_type": "nearby", "results_count": 0}
                )
            
            # Format result as readable text
            content = self._format_nearby_results(result)
            
            return Artifact(
                kind="text",
                content=content,
                meta={
                    "search_type": "nearby",
                    "results_count": result.get('total_found', 0),
                    "center": intent.center_location,
                    "target": intent.target_place,
                    "raw_data": result
                }
            )
        except Exception as e:
            print(f"[GEO_QUERY][ERROR] Nearby search failed: {e}")
            return Artifact(
                kind="text",
                content=f"Error during nearby search: {str(e)}",
                meta={"error": str(e)}
            )
    
    async def _handle_route_query(self, intent: GeoIntent) -> Artifact:
        """Handle route/directions query"""
        print(f"[GEO_QUERY][ROUTE] Finding route from '{intent.origin}' to '{intent.destination}'")
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.geo_tool.get_simple_transit_route,
                intent.origin,
                intent.destination,
                "LESS_WALKING"
            )
            
            if not result:
                return Artifact(
                    kind="text",
                    content=f"Unable to find route from {intent.origin} to {intent.destination}.",
                    meta={"search_type": "route", "success": False}
                )
            
            # Format route result
            content = self._format_route_result(result)
            
            return Artifact(
                kind="text",
                content=content,
                meta={
                    "search_type": "route",
                    "origin": intent.origin,
                    "destination": intent.destination,
                    "distance": result.get('distance'),
                    "duration": result.get('duration'),
                    "raw_data": result
                }
            )
        except Exception as e:
            print(f"[GEO_QUERY][ERROR] Route query failed: {e}")
            return Artifact(
                kind="text",
                content=f"Error finding route: {str(e)}",
                meta={"error": str(e)}
            )
    
    async def _handle_distance_query(self, intent: GeoIntent) -> Artifact:
        """Handle distance calculation query"""
        return Artifact(
            kind="text",
            content="Distance calculation feature is under development.",
            meta={"feature": "distance_calculation", "status": "not_implemented"}
        )
    
    def _format_nearby_results(self, result: dict) -> str:
        """Format nearby search results as readable text"""
        lines = [f"Found {result['total_found']} {result['target_place']} near {result['center_place']}:\n"]
        
        for i, place in enumerate(result.get('places', [])[:5], 1):
            lines.append(f"{i}. {place['name']}")
            lines.append(f"   Address: {place['address']}")
            if place.get('rating') and place['rating'] != 'No rating':
                lines.append(f"   Rating: {place['rating']}")
            lines.append(f"   Distance: {place['distance']}")
            if place.get('route'):
                lines.append(f"   Transit: {place['route']['distance']}, approximately {place['route']['duration']}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_route_result(self, result: dict) -> str:
        """Format route result as readable text"""
        lines = [
            f"Transit route from {result['origin']} to {result['destination']}:",
            f"Distance: {result['distance']}",
            f"Duration: {result['duration']}",
            f"Transfers: {result['steps_count']}\n",
            "Transit details:"
        ]
        
        for i, line in enumerate(result.get('transit_lines', []), 1):
            lines.append(f"  {i}. {line['vehicle']} {line['line']}")
            if line.get('from') and line.get('to'):
                lines.append(f"     From: {line['from']}")
                lines.append(f"     To: {line['to']}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _fallback_response(self, query: str) -> Artifact:
        """Fallback response when intent unclear"""
        return Artifact(
            kind="text",
            content=f"Sorry, I could not understand your geographic query: {query}\nPlease try queries like 'Find cafes near HKUST' or 'Route from CUHK to HKUST'",
            meta={"intent_unclear": True, "original_query": query}
        )
    
    async def _enhance_response(self, result: Artifact, ctx: Context) -> Artifact:
        """
        Enhance the response using LLM to make it more friendly and conversational.
        
        Args:
            result: Original artifact with raw geographic data
            ctx: Context with user query
            
        Returns:
            Enhanced artifact with friendlier content
        """
        print(f"[GEO_QUERY][ENHANCE] Starting enhancement...")
        
        try:
            from prompt import build_geo_query_instruction, get_length_hint
            from config_manager import get_config
            
            # Get response length config from config.yaml
            cfg = get_config()
            length_config = cfg.get_response_length_config("GEO_QUERY")
            max_tokens = length_config.get('max_tokens', 500)
            temperature = length_config.get('temperature', 0.3)
            
            # Use unified instruction hint builder
            instruction_hint = build_geo_query_instruction(ctx.query)
            instruction_hint += get_length_hint(max_tokens)
            
            # Prepare materials with original response
            materials = f"""# Original Route Information

{result.content}
"""
            
            # Use synthesizer.generate() for unified handling
            enhanced_content = await self.synthesizer.generate(
                category="GEO_QUERY",
                style_key="auto",
                constraints={
                    "language": "auto",
                    "tone": "friendly, conversational",
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "instruction_hint": instruction_hint
                },
                materials=materials,
                task_scaffold=None
            )
            
            print(f"[GEO_QUERY][ENHANCE] Completed (max_tokens={max_tokens})")
            
            # Create new artifact with enhanced content
            enhanced_result = Artifact(
                kind=result.kind,
                content=enhanced_content,
                meta={
                    **result.meta,
                    "enhanced": True,
                    "original_length": len(result.content),
                    "enhanced_length": len(enhanced_content),
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            
            return enhanced_result
            
        except Exception as e:
            print(f"[GEO_QUERY][ENHANCE][ERROR] Enhancement failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to original result if enhancement fails
            return result
        