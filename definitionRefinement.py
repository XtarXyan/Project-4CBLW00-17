import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx
import warnings

warnings.filterwarnings('ignore')

### What does this do?
#### This script analyzes specific neighborhoods/regions to compute lower bounds for 15-minute city classification.
#### The scores were extracted and noted down in eindhovenMap.py - this is because the ultimate result here still
#### required comparison with research papers I found online.

#### Optimal population density according to Peter Calthrope (considered an important figure in urban studies)
#### is around 8.000 people per km^2. For building density, I found a research paper titled 
#### "Sustainable urbanism: towards a framework for quality and optimal density?", from which we can outline the
#### 0.6-0.8 range for most transport-oriented "walkable cities."
#### The service diversity score is a bit more subjective, so it's mainly my own interpretation.

# Walking speed and time for 15-minute city (inclusive lower bound)
walk_speed_kph = 4
walk_time_min = 15
walk_radius_m = walk_speed_kph * 1000 / 60 * walk_time_min

# Amenity/service tags to use
tags = {
    'office': ['yes'],
    'amenity': [
        'pharmacy', 'hospital', 'clinic', 'doctors', 
        'school', 'kindergarten', 'college', 'university',  
        'cafe', 'restaurant', 'bar', 'cinema', 'theatre',  
        'community_centre', 'library', 'bicycle_rental',
        'place_of_worship', 
    ],
    'leisure': [
        'park', 'fitness_centre', 'sports_centre', 'stadium', 
        'dog_park', 'pitch', 'swimming_pool'
    ],
    'shop' : [
        'supermarket', 'convenience', 'bakery', 'greengrocer', 'butcher'
    ]
}

# Function to extract indicators for a neighborhood/region
def extract_neighborhood_indicators(neighborhood: str) -> dict:
    print(f"\nProcessing: {neighborhood}")

    try:
        # Load neighborhood boundary and walking network
        boundary = ox.geocode_to_gdf(neighborhood, which_result=1)
        area_km2 = boundary.geometry.area.iloc[0] / 1e6
        G = ox.graph_from_place(neighborhood, network_type='walk')
        G = ox.project_graph(G)

        # Load amenities
        gdf_amenities = ox.features_from_place(neighborhood, tags)
        gdf_amenities = gdf_amenities.to_crs(ox.graph_to_gdfs(G, nodes=True, edges=False).crs)

        # Walking accessibility
        center_node = ox.distance.nearest_nodes(G, boundary.geometry.centroid.x.iloc[0], boundary.geometry.centroid.y.iloc[0])
        subgraph = nx.ego_graph(G, center_node, radius=walk_radius_m, distance='length')
        nodes, _ = ox.graph_to_gdfs(subgraph)
        isochrone_poly = nodes.unary_union.convex_hull
        reachable_amenities = gdf_amenities[gdf_amenities.geometry.within(isochrone_poly)]
        walk_score = len(reachable_amenities)
        service_diversity = reachable_amenities[['amenity', 'shop', 'office', 'leisure']].nunique().sum()

        # Buildings
        buildings = ox.geometries_from_place(neighborhood, {'building': True})
        buildings = buildings.to_crs(boundary.crs)
        building_density = len(buildings) / area_km2

        # Population density
        if 'population' in boundary.columns:
            try:
                pop = int(boundary.population.iloc[0])
            except:
                pop = None
        else:
            pop = None

        if not pop:
            residential = buildings[buildings.get('building') == 'residential']
            pop = len(residential) * 2.5  # crude approximation

        pop_density = pop / area_km2

        return {
            'neighborhood': neighborhood,
            'walk_score': walk_score,
            'pop_density': round(pop_density, 2),
            'bldg_density': round(building_density, 2),
            'service_diversity': service_diversity
        }

    except Exception as e:
        print(f"Error processing {neighborhood}: {e}")
        return {
            'neighborhood': neighborhood,
            'walk_score': None,
            'pop_density': None,
            'bldg_density': None,
            'service_diversity': None
        }

# List of specific neighborhoods/regions to analyze
best_neighborhoods = [
    'Amsterdam, Netherlands',  # Broader query
    'Friedrichshain, Berlin, Germany',
    'Sunnyside, Portland, Oregon, USA',
    'Christianshavn, Copenhagen Municipality, Denmark',
    'El Poblenou, Barcelona, Spain',
    'Shimokitazawa, Tokyo, Japan',
    'Fitzroy, Melbourne, Australia'
]

# Analyze each neighborhood/region
results = [extract_neighborhood_indicators(neighborhood) for neighborhood in best_neighborhoods]

# Filter out neighborhoods with missing data
results = [result for result in results if None not in result.values()]

# Compute bounds for each property
def compute_bounds(data, keys):
    bounds = {}
    for key in keys:
        values = [d[key] for d in data]
        bounds[key] = (min(values), max(values))
    return bounds

keys = ['walk_score', 'pop_density', 'bldg_density', 'service_diversity']
bounds = compute_bounds(results, keys)

# Calculate lower bounds for 15-minute city classification
def calculate_lower_bounds(bounds, margin=0.1):
    """
    Calculate lower bounds for each property by taking the minimum value
    and applying a margin (e.g., 10% below the minimum).
    """
    lower_bounds = {}
    for key, (low, high) in bounds.items():
        lower_bounds[key] = max(0, low - (low * margin))  # Ensure no negative bounds
    return lower_bounds

lower_bounds = calculate_lower_bounds(bounds)

# Display results
print("\n--- Results ---")
for result in results:
    print(result)

print("\n--- Bounds ---")
for k, (low, high) in bounds.items():
    print(f"{k}: min = {low}, max = {high}")

print("\n--- Lower Bounds for 15-Minute City Classification ---")
for k, lb in lower_bounds.items():
    print(f"{k}: lower bound = {lb}")
