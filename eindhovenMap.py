import osmnx as ox
import geopandas as gpd
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import folium
from shapely.geometry import Point, mapping
from concurrent.futures import ThreadPoolExecutor

# Enable caching for OSM queries
ox.settings.use_cache = True

# Define the maximum number of improvements for low-scoring zones
MAX_IMPROVEMENTS = 2

#### PLEASE READ FIRST!
#### Version 2.0
## Originally, this was supposed to be a script to check for the current separation of
## neighbourhoods in Eindhoven, and if they fit the classification of a 15-minute city,
## so we could recognize districts that we could immediately reject. However, that's not
## really necessary anymore, so instead this will use Fieke's definitions of a 15-minute
## city, as based on the bounds extracted in definitionRefinement.py, and will try to
## separate Eindhoven into 15-minute zones that way.

## Another thing to keep in mind are the properties for the building density, population
## density, and service diversity. These are all based on a simulation provided in the other
## script, and the values here are defined just to minimize running times, though running
## with the simulated ones works fine just for checking which neighbourhoods are satisfactory.
## OPTIMAL VALUES:
## Population density: 8631 - 12400 people/km2
## Building density: 0.6 - 0.8 built zone per total area
## Service diversity: > 1.5  different services/km2 // Maximum extracted was around 20 services per km^2

## This is a work in progress, but is almost done. The only thing that needs modifying now,
## is the full loop; this simulation is unfortunately quite time-consuming, since the model
## needs to run enough times until it stops finding "low" scoring zones.
## By debugging with smaller walking radii, the code seems to function properly.

# Speeds remain the same. Comments left for clarity.

place_name = "Eindhoven, Netherlands"

# According to the Journal of Applied Physiology, average walking speed falls 
# between 4.0 - 5.9 km/h. Here, we use the lower bound, since the model is supposed
# to be as inclusive as possible.
average_walk_speed = 4

# It's a 15-minute city, so:
time_min = 15

walk_radius_m = average_walk_speed * 1000 / 60 * time_min    

# Tags remain the same as in original version, based on Moreno's definition; will be
# updated if necessary, still subject to change.

tags = {
    'office': ['yes'],  # All workplaces, since OSM has an incredible amount of identifiers.
    'shop': ['supermarket', 'convenience', 'bakery', 'greengrocer', 'butcher'],
    'amenity': [
        'pharmacy', 'hospital', 'clinic', 'doctors', 
        'school', 'kindergarten', 'college', 'university',  
        'cafe', 'restaurant', 'bar', 'cinema', 'theatre',  
        'community_centre', 'library', 'bicycle_rental',
        'place_of_worship', 
        # Last one I'm unsure of, especially since this is a "diverse" business plan? But nonetheless.
    ],
    'leisure': [
        'park', 'fitness_centre', 'sports_centre', 'stadium', 
        'dog_park', 'pitch', 'swimming_pool'
    ]
    # Not sure how many leisures to include, so included the "green" ones.
}

# tags = {
#     'office': ['yes'],
#     'shop': ['yes'],
#     'amenity': ['yes'],
#     'leisure': ['yes'],
# }

# I still need to extract the specific property bounds from the definitionRefinement.py script,
# and compare them to some research papers to check the research's validity. For now, I will 
# just use some generalized values that I got from asking Copilot - thanks, Copilot. 
    
# Population density is measured here using people per km2.
POP_DENSITY_MIN = 4000
POP_DENSITY_MAX = 10000
# Building density is meaured here using percentage of built area to total area.
BUILDING_DENSITY_MIN = 0.3
BUILDING_DENSITY_MAX = 0.7
# Service diversity is measured here using the number of different services per km2 using
# Shannon entropy. Apparently the standard. I have no idea.
SERVICE_DIVERSITY_MIN = 1.5

# Big change: Eindhoven's admin levels are not there in OSM, only the adminsitrative boundaries
# for Noord-Brabant, and so it's impossible to use the admin_level to extract the city. Instead,
# I use the geopackage provided by the Central Bureau voor de Statistiek (CBS.)

# First, we will extract the city + buurten boundaries from the geopackage, and plot these.
buurten = gpd.read_file("wijkenbuurten_2023_v2.gpkg", layer="wijken")
eindhoven_buurten = buurten[buurten["gemeentenaam"] == "Eindhoven"]
eindhoven_buurten = eindhoven_buurten.to_crs(epsg=3857)  # for distance in meters

eindhoven_buurten["centroid"] = eindhoven_buurten.geometry.centroid

fig, ax = plt.subplots(figsize=(10, 10))
eindhoven_buurten.plot(edgecolor='black', column='wijknaam', cmap='tab20')
# plt.show()

# This gives us a nice overview of the neighbourhoods in Eindhoven, which we can later use 
# for basing our 15-minute walkability inquiry. We will be modifying these in accordance with
# the properties (POP_DENSITY, BUILDING_DENSITY, SERVICE_DIVERSITY) to check which neighbourhoods
# satisfy the properties, and then modify their boundaries accordingly.

G = ox.graph_from_place(place_name, network_type="walk", simplify=True)

# This is awfully smart - an EGO GRAPH is one that centers around some node - we will define this
# ego node to be the node located at the centroid of the neighbourhood. The ego graph will then be a subgraph
# of the original graph, containing all nodes within the specified radius.

def make_isochrone(point, G, walk_dist = 1200):
    node = ox.nearest_nodes(G, point.x, point.y)    
    subgraph = nx.ego_graph(G, node, radius=walk_dist, distance='length')
    # Polygon from subgraph nodes
    nodes = [G.nodes[n] for n in subgraph.nodes]
    points = [Point(data['x'], data['y']) for data in nodes]
    polygon = gpd.GeoSeries(points).union_all().convex_hull
    return polygon

# A beautiful solution suggested by Stackexchange: parallelization of isochrone generation saves
# minutes of running time.
def generate_isochrone(row):
    return make_isochrone(row.centroid, G)

with ThreadPoolExecutor() as executor:
    isochrones = list(executor.map(generate_isochrone, eindhoven_buurten.itertuples()))
eindhoven_buurten["isochrone_15min"] = isochrones

# Let's define points of interest as specific buurtens of Eindhoven for further analysis.
POIS = ox.features_from_place(place_name, tags)
POIS = POIS.to_crs(eindhoven_buurten.crs)

# Method for classifying different types of services; need it so that later when defining 
# "scoring" for the services we can do it based on the type of service.
def classify_service(row):
    return row.get('amenity') or row.get('shop') or row.get('leisure') or row.get('office')

POIS["service_type"] = POIS.apply(classify_service, axis=1)
pois = POIS[~POIS["service_type"].isna()]

# The two methods below are very simple, just needed so that we can check whether or not Fieke's
# criteria are satisfied by the newly proposed designations.

def count_services(polygon, pois):
    return pois[pois.geometry.intersects(polygon)]

def shannon_diversity(services):
    counts = services["service_type"].value_counts()
    proportions = counts / counts.sum()
    return -(proportions * np.log(proportions)).sum()

# This is the main scoring loop, where we check the properties of each neighbourhood
# and assign a score based on the properties. The scoring is done based on the properties
# defined above, and how well each defined isochrone fits their boundaries.

results = []
for _, row in eindhoven_buurten.iterrows():
    poly = row["isochrone_15min"]
    services = count_services(poly, pois)
    service_diversity = shannon_diversity(services) if len(services) > 0 else 0
    pop_density = row["omgevingsadressendichtheid"]
    building_density = np.random.uniform(0.2, 0.8)  # Placeholder

    passed = {
        "population": POP_DENSITY_MIN <= pop_density <= POP_DENSITY_MAX,
        "building": BUILDING_DENSITY_MIN <= building_density <= BUILDING_DENSITY_MAX,
        "diversity": service_diversity >= SERVICE_DIVERSITY_MIN,
    }

    if all(passed.values()):
        score = "high"
    elif any(passed.values()):
        score = "medium"
    else:
        score = "low"

    results.append({
        "service_count": len(services),
        "service_diversity": service_diversity,
        "building_density": building_density,
        "score": score
    })

results_df = pd.DataFrame(results)
eindhoven_buurten = pd.concat([eindhoven_buurten.reset_index(drop=True), results_df], axis=1)

# Expand the zones that scored "low" to see if we can improve their scores by expanding the isochrone. 
# Note that this only improves them  MAX_IMPROVEMENTS iterations.
def try_expand(row, G, pois, walk_dist=1800, max_improvements=MAX_IMPROVEMENTS):
    if row["score"] != "low":
        return row["isochrone_15min"], row["score"]

    polygon = row["isochrone_15min"]
    score = "low"

    for _ in range(max_improvements):
        node = ox.nearest_nodes(G, row["centroid"].x, row["centroid"].y)
        subgraph = nx.ego_graph(G, node, radius=walk_dist, distance='length')
        nodes = [G.nodes[n] for n in subgraph.nodes]
        points = [Point(data['x'], data['y']) for data in nodes]
        polygon = gpd.GeoSeries(points).union_all().convex_hull
        services = count_services(polygon, pois)
        service_diversity = shannon_diversity(services) if len(services) > 0 else 0
        pop_density = row["omgevingsadressendichtheid"]
        building_density = np.random.uniform(0.2, 0.8)

        if (
            POP_DENSITY_MIN <= pop_density <= POP_DENSITY_MAX and
            BUILDING_DENSITY_MIN <= building_density <= BUILDING_DENSITY_MAX and
            service_diversity >= SERVICE_DIVERSITY_MIN
        ):
            score = "upgraded"
            break

    return polygon, score

expanded = eindhoven_buurten.apply(lambda row: try_expand(row, G, pois), axis=1)
eindhoven_buurten["final_isochrone"] = expanded.apply(lambda x: x[0])
eindhoven_buurten["final_score"] = expanded.apply(lambda x: x[1])

# --- Plotting to Folium ---
m = folium.Map(location=[51.44, 5.48], zoom_start=12, tiles="cartodbpositron")
color_map = {"high": "green", "medium": "orange", "low": "red", "upgraded": "blue"}

for _, row in eindhoven_buurten.iterrows():
    if not row["final_isochrone"].is_empty:
        folium.GeoJson(
            mapping(row["final_isochrone"]),
            style_function=lambda feature, color=color_map[row["final_score"]]: {
                "fillColor": color,
                "color": color,
                "weight": 1,
                "fillOpacity": 0.4,
            },
            tooltip=f"{row['wijknaam']} ({row['final_score']})"
        ).add_to(m)

m.save("eindhoven_final_15min_map.html")
m