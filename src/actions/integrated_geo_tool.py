import requests
import json
import math
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class IntegratedGeoTool:
    def __init__(self, api_key=None):
        # Use provided API key or get from environment variable
        self.api_key = api_key or os.getenv('GOOGLE_MAP_KEY')
        if not self.api_key:
            raise ValueError("Google Maps API key not found. Please set GOOGLE_MAP_KEY environment variable or provide api_key parameter.")
        
        self.places_base_url = "https://places.googleapis.com/v1/places"
        self.routes_base_url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    
    def calculate_distance(self, lat1, lng1, lat2, lng2):
        """Calculate distance between two coordinates using Haversine formula (in meters)"""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return round(distance, 1)
    
    def get_place_coordinates(self, place_name):
        """Get coordinates of a place by name"""
        url = f"{self.places_base_url}:searchText"
        
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': self.api_key,
            'X-Goog-FieldMask': 'places.displayName,places.location,places.formattedAddress'
        }
        
        data = {
            "textQuery": place_name,
            "languageCode": "zh-CN",
            "maxResultCount": 1
        }
        
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        
        if 'places' in result and len(result['places']) > 0:
            location = result['places'][0]['location']
            return location['latitude'], location['longitude']
        return None, None
    
    def search_nearby_places(self, center_place, target_place, radius):
        """Search for target places near a center location using text search with distance sorting"""
        # Step 1: Get center place coordinates
        lat, lng = self.get_place_coordinates(center_place)
        
        if lat is None or lng is None:
            return {"error": f"Unable to find location: {center_place}"}
        
        print(f"Found {center_place} coordinates: {lat}, {lng}")
        
        # Step 2: Use text search to find target places near the center location
        url = f"{self.places_base_url}:searchText"
        
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': self.api_key,
            'X-Goog-FieldMask': 'places.displayName,places.formattedAddress,places.location,places.rating,places.types'
        }
        
        # Build search query
        search_text = f"{target_place} near {center_place}"
        
        data = {
            "textQuery": search_text,
            "languageCode": "zh-CN",
            "maxResultCount": 20,
            "locationBias": {
                "circle": {
                    "center": {
                        "latitude": lat,
                        "longitude": lng
                    },
                    "radius": radius
                }
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        
        # Error handling
        if response.status_code != 200:
            print(f"API request failed: {response.status_code}")
            print(f"Error message: {result}")
            return {"error": f"API request failed: {result.get('message', 'Unknown error')}"}
        
        if 'places' in result:
            print(f"Found {len(result['places'])} relevant places")
            
            # Calculate distance for each place
            for place in result['places']:
                place_location = place.get('location', {})
                place_lat = place_location.get('latitude')
                place_lng = place_location.get('longitude')
                
                if place_lat and place_lng:
                    distance = self.calculate_distance(lat, lng, place_lat, place_lng)
                    place['distance_meters'] = distance
                else:
                    place['distance_meters'] = float('inf')
            
            # Sort by distance (nearest first)
            result['places'].sort(key=lambda x: x.get('distance_meters', float('inf')))
            print("Sorted by distance (nearest first)")
        
        return result
    
    def decode_polyline(self, encoded_polyline):
        """Decode Google polyline to list of coordinates"""
        points = []
        index = 0
        lat = 0
        lng = 0
        
        while index < len(encoded_polyline):
            # Decode latitude
            result = 1
            shift = 0
            while True:
                b = ord(encoded_polyline[index]) - 63 - 1
                index += 1
                result += b << shift
                shift += 5
                if b < 0x1f:
                    break
            lat += (~result >> 1) if (result & 1) != 0 else (result >> 1)
            
            # Decode longitude
            result = 1
            shift = 0
            while True:
                b = ord(encoded_polyline[index]) - 63 - 1
                index += 1
                result += b << shift
                shift += 5
                if b < 0x1f:
                    break
            lng += (~result >> 1) if (result & 1) != 0 else (result >> 1)
            
            points.append([lat / 1e5, lng / 1e5])
        
        return points
    
    def get_route(self, origin_lat, origin_lng, dest_lat, dest_lng, travel_mode="TRANSIT"):
        """Get route between two coordinates"""
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': self.api_key,
            'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline'
        }
        
        data = {
            "origin": {
                "location": {
                    "latLng": {
                        "latitude": origin_lat,
                        "longitude": origin_lng
                    }
                }
            },
            "destination": {
                "location": {
                    "latLng": {
                        "latitude": dest_lat,
                        "longitude": dest_lng
                    }
                }
            },
            "travelMode": travel_mode,
            "languageCode": "zh-CN",
            "units": "METRIC"
        }
        
        response = requests.post(self.routes_base_url, headers=headers, json=data)
        return response.json()
    
    def get_transit_route_by_address(self, origin_address, dest_address, routing_preference=None, allowed_modes=None):
        """Get the fastest transit route between two addresses"""
        if allowed_modes is None:
            allowed_modes = ["TRAIN", "BUS", "SUBWAY"]
        
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': self.api_key,
            'X-Goog-FieldMask': 'routes.legs.steps.transitDetails,routes.legs.steps.staticDuration,routes.legs.steps.travelMode,routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline'
        }
        
        data = {
            "origin": {
                "address": origin_address
            },
            "destination": {
                "address": dest_address
            },
            "travelMode": "TRANSIT",
            "computeAlternativeRoutes": False,  # Only get the best route
            "transitPreferences": {
                "routingPreference": routing_preference,
                "allowedTravelModes": allowed_modes
            },
            "languageCode": "zh-CN",
            "units": "METRIC"
        }
        
        response = requests.post(self.routes_base_url, headers=headers, json=data)
        result = response.json()
        
        # If address-based query fails, try with coordinates
        if 'error' in result and result['error'].get('status') == 'NOT_FOUND':
            print(f"Address not found, trying with coordinates...")
            
            # Get coordinates for both addresses
            origin_lat, origin_lng = self.get_place_coordinates(origin_address)
            dest_lat, dest_lng = self.get_place_coordinates(dest_address)
            
            if origin_lat and dest_lat:
                # Retry with coordinates
                data = {
                    "origin": {
                        "location": {
                            "latLng": {
                                "latitude": origin_lat,
                                "longitude": origin_lng
                            }
                        }
                    },
                    "destination": {
                        "location": {
                            "latLng": {
                                "latitude": dest_lat,
                                "longitude": dest_lng
                            }
                        }
                    },
                    "travelMode": "TRANSIT",
                    "computeAlternativeRoutes": False,
                    "transitPreferences": {
                        "routingPreference": routing_preference,
                        "allowedTravelModes": allowed_modes
                    },
                    "languageCode": "zh-CN",
                    "units": "METRIC"
                }
                
                response = requests.post(self.routes_base_url, headers=headers, json=data)
                result = response.json()
                print(f"Coordinate-based query result: {list(result.keys())}")
        
        return result

        # return response.json()
    
    def analyze_transit_route(self, route_data):
        """Analyze the best transit route (first route only)"""
        if 'routes' not in route_data or not route_data['routes']:
            return None
        
        # Only analyze the first (best) route
        route = route_data['routes'][0]
        
        route_info = {
            'distance_km': route['distanceMeters'] / 1000,
            'duration_seconds': int(route['duration'][:-1]),
            'duration_minutes': int(route['duration'][:-1]) // 60,
            'steps': [],
            'coordinates': []
        }
        
        # Decode polyline
        if 'polyline' in route and 'encodedPolyline' in route['polyline']:
            encoded = route['polyline']['encodedPolyline']
            route_info['coordinates'] = self.decode_polyline(encoded)
        
        # Analyze transit steps - only keep transit steps
        if 'legs' in route:
            for leg in route['legs']:
                if 'steps' in leg:
                    for step in leg['steps']:
                        # Only include steps with transit details
                        if 'transitDetails' in step:
                            transit = step['transitDetails']
                            step_info = {
                                'travel_mode': step.get('travelMode', 'TRANSIT'),
                                'line_name': transit.get('transitLine', {}).get('name', ''),
                                'line_short_name': transit.get('transitLine', {}).get('nameShort', ''),
                                'vehicle_type': transit.get('transitLine', {}).get('vehicle', {}).get('type', ''),
                                'departure_stop': transit.get('stopDetails', {}).get('departureStop', {}).get('name', ''),
                                'arrival_stop': transit.get('stopDetails', {}).get('arrivalStop', {}).get('name', ''),
                                'num_stops': transit.get('stopCount', 0),
                                'headsign': transit.get('headsign', ''),
                                'duration_seconds': int(step.get('staticDuration', '0s')[:-1]) if step.get('staticDuration') else 0
                            }
                            route_info['steps'].append(step_info)
        
        return route_info
    
    def print_transit_details(self, route_info, origin_address, dest_address):
        """Print simplified transit route information (best route only)"""
        if not route_info:
            print("No transit route found")
            return
        
        #print(f"\n{'='*60}")
        #print(f"BEST TRANSIT ROUTE: {origin_address} → {dest_address}")
        #print(f"{'='*60}")
        
        #print(f"Total distance: {route_info['distance_km']:.2f}km")
        #print(f"Total time: {route_info['duration_minutes']}min {route_info['duration_seconds']%60}sec")
        #print(f"Transit steps: {len(route_info['steps'])}")
        
        if route_info['steps']:
            print(f"\nTransit Details:")
            for i, step in enumerate(route_info['steps'], 1):
                line_display = step['line_name']
                if step['line_short_name']:
                    line_display += f" ({step['line_short_name']})"
                
                vehicle_icon = "🚇" if step['vehicle_type'] in ['SUBWAY', 'TRAIN'] else "🚌"
                
                print(f"  {i}. {vehicle_icon} {line_display}")
                
                if step['departure_stop'] and step['arrival_stop']:
                    print(f"     🚏 {step['departure_stop']} → {step['arrival_stop']}")
                
                if step['num_stops'] > 0:
                    print(f"     📍 {step['num_stops']} stops")
                
                if step['headsign']:
                    print(f"     🎯 Direction: {step['headsign']}")
                
                if step['duration_seconds'] > 0:
                    mins = step['duration_seconds'] // 60
                    secs = step['duration_seconds'] % 60
                    if mins > 0:
                        print(f"     ⏱️ {mins}min {secs}sec")
                    else:
                        print(f"     ⏱️ {secs}sec")
                
                print()
        
        return route_info
    
    def analyze_route(self, route_data):
        """Analyze and display route information"""
        if 'routes' not in route_data or not route_data['routes']:
            return None
        
        route = route_data['routes'][0]
        
        route_info = {
            'distance_km': route['distanceMeters'] / 1000,
            'duration_seconds': int(route['duration'][:-1]),
            'duration_minutes': int(route['duration'][:-1]) // 60,
            'coordinates': []
        }
        
        # Decode polyline
        if 'polyline' in route and 'encodedPolyline' in route['polyline']:
            encoded = route['polyline']['encodedPolyline']
            route_info['coordinates'] = self.decode_polyline(encoded)
        
        return route_info
    
    def create_route_map_html(self, coordinates, start_name, end_name, filename="route_map.html"):
        # Ensure output directory exists
        import os
        output_dir = "output/maps"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create full path for the file
        if not filename.startswith("output/"):
            filename = os.path.join(output_dir, filename)
        """Create a simple HTML map to visualize the route"""
        if not coordinates:
            return False
        
        center_lat = sum(coord[0] for coord in coordinates) / len(coordinates)
        center_lng = sum(coord[1] for coord in coordinates) / len(coordinates)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Route: {start_name} to {end_name}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
</head>
<body>
    <h2>Route: {start_name} → {end_name}</h2>
    <div id="map" style="width: 100%; height: 600px;"></div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lng}], 13);
        
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        var coordinates = {coordinates};
        var polyline = L.polyline(coordinates, {{color: 'red', weight: 4}}).addTo(map);
        
        // Add start and end markers
        L.marker([{coordinates[0][0]}, {coordinates[0][1]}]).addTo(map)
            .bindPopup('{start_name}').openPopup();
        L.marker([{coordinates[-1][0]}, {coordinates[-1][1]}]).addTo(map)
            .bindPopup('{end_name}');
        
        map.fitBounds(polyline.getBounds());
    </script>
</body>
</html>
"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return True
    
    def find_places_with_routes(self, center_place, target_place, radius=2000, travel_mode="TRANSIT", max_results=5):
        """Complete workflow: find places and get routes to them"""
        print(f"=== Searching for {target_place} near {center_place} ===")
        
        # Step 1: Find nearby places
        places_result = self.search_nearby_places(center_place, target_place, radius)
        
        if 'error' in places_result:
            print(f"Error: {places_result['error']}")
            return None
        
        if 'places' not in places_result or not places_result['places']:
            print(f"No {target_place} found near {center_place}")
            return None
        
        # Step 2: Get center coordinates for route calculation
        center_lat, center_lng = self.get_place_coordinates(center_place)
        
        results = []
        places_to_process = places_result['places'][:max_results]
        
        print(f"\n=== Found {len(places_to_process)} {target_place} locations ===")
        
        for i, place in enumerate(places_to_process, 1):
            name = place.get('displayName', {}).get('text', 'Unknown')
            address = place.get('formattedAddress', 'Address unknown')
            rating = place.get('rating', 'No rating')
            distance = place.get('distance_meters', 0)
            
            place_location = place.get('location', {})
            place_lat = place_location.get('latitude')
            place_lng = place_location.get('longitude')
            
            print(f"\n{i}. {name}")
            print(f"   Address: {address}")
            print(f"   Rating: {rating}")
            if distance < 1000:
                print(f"   Distance: {distance}m")
            else:
                print(f"   Distance: {distance/1000:.1f}km")
            
            # Get route information
            if place_lat and place_lng:
                print(f"   Getting {travel_mode.lower()} route...")
                route_data = self.get_route(center_lat, center_lng, place_lat, place_lng, travel_mode)
                route_info = self.analyze_route(route_data)
                
                if route_info:
                    print(f"   Route distance: {route_info['distance_km']:.2f}km")
                    print(f"   Route time: {route_info['duration_minutes']}min {route_info['duration_seconds']%60}sec")
                    
                    # Create individual map for this route
                    # if route_info['coordinates']:
                    #    map_filename = f"output/maps/route_to_{name.replace(' ', '_').replace('/', '_')}.html"
                    #    if self.create_route_map_html(route_info['coordinates'], center_place, name, map_filename):
                    #        print(f"   Route map: {map_filename}")
                    map_filename = None
                    results.append({
                        'place': place,
                        'route_info': route_info,
                        'map_file': map_filename if route_info['coordinates'] else None
                    })
                else:
                    print(f"   No route available")
            else:
                print(f"   No coordinates available")
        
        return results
    
    def find_places_simple(self, center_place, target_place, radius=2000, travel_mode="TRANSIT", max_results=5):
        """Simplified version of find_places_with_routes for LLM output"""
        # Get full results
        full_results = self.find_places_with_routes(center_place, target_place, radius, travel_mode, max_results)
        
        if not full_results:
            return None
        
        # Create simplified summary
        summary = {
            'center_place': center_place,
            'target_place': target_place,
            'total_found': len(full_results),
            'places': []
        }
        
        for result in full_results:
            place = result['place']
            route_info = result.get('route_info')
            
            place_summary = {
                'name': place.get('displayName', {}).get('text', 'Unknown'),
                'address': place.get('formattedAddress', 'Address unknown'),
                'rating': place.get('rating', 'No rating'),
                'distance_meters': place.get('distance_meters', 0)
            }
            
            # Add distance in human-readable format
            if place_summary['distance_meters'] < 1000:
                place_summary['distance'] = f"{place_summary['distance_meters']}m"
            else:
                place_summary['distance'] = f"{place_summary['distance_meters']/1000:.1f}km"
            
            # Add route information if available (without coordinates)
            if route_info:
                place_summary['route'] = {
                    'distance': f"{route_info['distance_km']:.1f}km",
                    'duration': f"{route_info['duration_minutes']}分钟"
                }
            
            summary['places'].append(place_summary)
        
        return summary
    
    def get_best_transit_route(self, origin_address, dest_address, routing_preference="LESS_WALKING", allowed_modes=None):
        """Get the best (fastest) transit route between two addresses"""
        print(f"[GEO_TOOL] Getting transit route from {origin_address} to {dest_address}")
        
        # Get transit route data
        route_data = self.get_transit_route_by_address(
            origin_address, 
            dest_address, 
            routing_preference, 
            allowed_modes
        )
        
        # Analyze the best route
        route_info = self.analyze_transit_route(route_data)
        
        # if route_info:
            # Silent mode: don't print details here, let the caller format the output
            # self.print_transit_details(route_info, origin_address, dest_address)
            
            # Create map for the best route
            #if route_info['coordinates']:
            #    map_filename = f"output/maps/best_transit_route_{origin_address.replace(' ', '_')}_to_{dest_address.replace(' ', '_')}.html"
            #    if self.create_route_map_html(
            #        route_info['coordinates'], 
            #        origin_address, 
            #        dest_address, 
            #        map_filename
            #    ):
            #        print(f"[GEO_TOOL] Route map created: {map_filename}")
        
        return route_info
    
    def get_simple_transit_route(self, origin_address, dest_address, routing_preference="LESS_WALKING", allowed_modes=None):
        """Get simplified transit route information for LLM output"""
        # Get full route info
        route_info = self.get_best_transit_route(origin_address, dest_address, routing_preference, allowed_modes)
        
        if not route_info:
            return None
        
        # Create simplified summary for LLM
        summary = {
            'origin': origin_address,
            'destination': dest_address,
            'distance': f"{route_info['distance_km']:.1f}km",
            'duration': f"{route_info['duration_minutes']}分钟",
            'steps_count': len(route_info['steps']),
            'transit_lines': []
        }
        
        # Extract key transit information
        for step in route_info['steps']:
            line_info = step['line_name']
            if step['line_short_name']:
                line_info += f" ({step['line_short_name']})"
            
            vehicle_icon = "🚇" if step['vehicle_type'] in ['SUBWAY', 'TRAIN'] else "🚌"
            
            # Only include lines with meaningful names
            if line_info.strip():
                summary['transit_lines'].append({
                    'line': line_info,
                    'vehicle': vehicle_icon,
                    'from': step.get('departure_stop', ''),
                    'to': step.get('arrival_stop', '')
                })
        
        return summary
    
    def get_driving_route(self, origin_address, dest_address):
        """Get driving route between two addresses"""
        print(f"=== Getting driving route from {origin_address} to {dest_address} ===")
        
        # Get coordinates for both addresses
        origin_lat, origin_lng = self.get_place_coordinates(origin_address)
        dest_lat, dest_lng = self.get_place_coordinates(dest_address)
        
        if origin_lat is None or dest_lat is None:
            print("Unable to find coordinates for one or both addresses")
            return None
        
        print(f"Origin: {origin_lat}, {origin_lng}")
        print(f"Destination: {dest_lat}, {dest_lng}")
        
        # Get driving route
        route_data = self.get_route(origin_lat, origin_lng, dest_lat, dest_lng, "DRIVE")
        route_info = self.analyze_route(route_data)
        
        if route_info:
            print(f"\n🚗 DRIVING ROUTE: {origin_address} → {dest_address}")
            print(f"Distance: {route_info['distance_km']:.2f}km")
            print(f"Time: {route_info['duration_minutes']}min {route_info['duration_seconds']%60}sec")
            
            # Create map for driving route
            # if route_info['coordinates']:
            #    map_filename = f"output/maps/driving_route_{origin_address.replace(' ', '_')}_to_{dest_address.replace(' ', '_')}.html"
            #    if self.create_route_map_html(
            #        route_info['coordinates'], 
            #        origin_address, 
            #        dest_address, 
            #        map_filename
            #    ):
            #        print(f"Driving route map created: {map_filename}")
        
        return route_info

def print_summary(results, center_place, target_place):
    """Print a summary of all results"""
    if not results:
        return
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {target_place} near {center_place}")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        place = result['place']
        route_info = result['route_info']
        name = place.get('displayName', {}).get('text', 'Unknown')
        distance = place.get('distance_meters', 0)
        
        print(f"{i}. {name}")
        print(f"   Straight-line distance: {distance}m" if distance < 1000 else f"   Straight-line distance: {distance/1000:.1f}km")
        if route_info:
            print(f"   Route distance: {route_info['distance_km']:.2f}km")
            print(f"   Travel time: {route_info['duration_minutes']}min {route_info['duration_seconds']%60}sec")
        print()

# Example usage and test
if __name__ == "__main__":
    # Initialize without API key - will use environment variable
    geo_tool = IntegratedGeoTool()
    
    print("🎯 INTEGRATED GEO TOOL TEST")
    print("="*50)
    
    # Test 1: Find places with routes
    print("\n📍 TEST 1: Find KFC near HKUST with routes")
    center_place = "将军澳"
    target_place = "donki"
    radius = 3000  # 3km radius
    travel_mode = "TRANSIT"  # Public transport
    
    results = geo_tool.find_places_with_routes(
        center_place=center_place,
        target_place=target_place,
        radius=radius,
        travel_mode=travel_mode,
        max_results=3
    )
    
    print_summary(results, center_place, target_place)
    
    # Test 2: Best transit route between universities
    print("\n🚇 TEST 2: Best transit route between universities")
    origin_address = "香港中文大学"
    dest_address = "香港科技大学"
    
    transit_route = geo_tool.get_best_transit_route(
        origin_address=origin_address,
        dest_address=dest_address,
        routing_preference="LESS_WALKING",
        allowed_modes=["TRAIN", "BUS", "SUBWAY"]
    )
    
    print(f"\n✅ Test completed! Found {'1 best' if transit_route else 'no'} transit route")
    
    # Test 3: Driving route between universities
    print("\n🚗 TEST 3: Driving route between universities")
    
    driving_route = geo_tool.get_driving_route(
        origin_address="香港中文大学",
        dest_address="香港科技大学"
    )
    
    print(f"\n✅ Driving test completed! Found {'1' if driving_route else 'no'} driving route") 