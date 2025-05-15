import osmnx as ox
import geopandas as gpd
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import folium
from shapely.geometry import Point, mapping
from shapely.ops import unary_union
from concurrent.futures import ThreadPoolExecutor

# Enable caching for OSM queries: we need this so that pulling data does not lead to a timeout.
ox.settings.use_cache = True
ox.settings.timeout = 600  # Set timeout to 300 seconds
ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api/"


#### PLEASE READ FIRST!
#### Version 2.0
## Originally, this was supposed to be a script to check for the current separation of
## neighbourhoods in Eindhoven, and if they fit the classification of a 15-minute city,
## so we could recognize districts that we could immediately reject. However, that's not
## really necessary anymore, so instead this will use Fieke's definitions of a 15-minute
## city, as based on the bounds extracted in definitionRefinement.py, and will try to
## separate Eindhoven into 15-minute zones that way.

### NOTE ABOUT THIS ONE
## THIS VERSION ONLY USES POPULATION DENSITY, AND CONTAINS A LOT OF DEBUGGING COMMENTS.
## This is because building density is almost impossible to acquire for some reason.

## Another thing to keep in mind are the properties for the building density, population
## density, and service diversity. These are all based on a simulation provided in the other
## script, and the values here are defined just to minimize running times, though running
## with the simulated ones works fine just for checking which neighbourhoods are satisfactory.
## OPTIMAL VALUES:
## Population density for large city: 8631 - 12400 people/km2
## Standardized for a smaller size like Eindhoven: [10000-14400]*0.161 = [1610 - 2318] 
## --> Higher bound moved to include larger inner density to ~ 4632
## Standardized for intermediate city size: 3500 - 8000 people/km2
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
    ],
    'shop' : [
        'supermarket', 'convenience', 'bakery', 'greengrocer', 'butcher'
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
print('Enter minimum population density:')
print('Low-density: 1610+ | Medium-density: 4632+ | High-density: 8631+ (not recommended)')
minDensityInput = int(input())
print('Enter maximum population density:')
print('Recommended at least twice minimum bound size (or standard of 6000).')
maxDensityInput = int(input())

POP_DENSITY_MIN = minDensityInput
POP_DENSITY_MAX = maxDensityInput

print('Loading network from CBS file...this may take a while')

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

# This gives us a nice overview of the neighbourhoods in Eindhoven, which we can later use 
# for basing our 15-minute walkability inquiry. We will be modifying these in accordance with
# the properties (POP_DENSITY, BUILDING_DENSITY, SERVICE_DIVERSITY) to check which neighbourhoods
# satisfy the properties, and then modify their boundaries accordingly.

G = ox.graph_from_place(place_name, network_type="walk", simplify=True)

# This is awfully smart - an EGO GRAPH is one that centers around some node - we will define this
# ego node to be the node located at the centroid of the neighbourhood. The ego graph will then be a subgraph
# of the original graph, containing all nodes within the specified radius.

def make_isochrone(point, G, walk_dist = walk_radius_m):
    node = ox.nearest_nodes(G, point.x, point.y)    
    subgraph = nx.ego_graph(G, node, radius=walk_dist, distance='length')
    nodes = [G.nodes[n] for n in subgraph.nodes]
    points = [Point(data['x'], data['y']) for data in nodes]
    polygon = gpd.GeoSeries(points).unary_union.convex_hull
    return polygon

# # A beautiful solution suggested by Stackexchange: parallelization of isochrone generation saves
# # minutes of running time.
def generate_isochrone(row):
    return make_isochrone(row.centroid, G)
with ThreadPoolExecutor() as executor:
    isochrones = list(executor.map(generate_isochrone, eindhoven_buurten.itertuples()))
eindhoven_buurten["isochrone_15min"] = isochrones

# Let's define points of interest as specific buurtens of Eindhoven for further analysis.
POIS = ox.features_from_place(place_name, tags)
POIS = POIS.to_crs(eindhoven_buurten.crs)

# Method for classifying services; need it so that later when defining 
# "scoring" for the services we can do it based on the type of service.
def classify_service(row):
    return row.get('amenity') or row.get('shop') or row.get('leisure') or row.get('office')


POIS["service_type"] = POIS.apply(classify_service, axis=1)
pois = POIS[~POIS["service_type"].isna()]

# This is the main scoring loop, where we check the properties of each neighbourhood
# and assign a score based on the properties. The scoring is done based on the properties
# defined above, and how well each defined isochrone fits their boundaries.

results = []
for _, row in eindhoven_buurten.iterrows():
    poly = row["isochrone_15min"]
    pop_density = row["omgevingsadressendichtheid"]

    passed = {
        "population": POP_DENSITY_MIN <= pop_density <= POP_DENSITY_MAX,
    }

    if all(passed.values()):
        score = "pass"
    else:
        score = "fail"

    results.append({
        "wijknaam": row["wijknaam"],
        "geometry": row["geometry"],
        "isochrone": poly,
        "score": score,
        "pop_density": pop_density,
    })


results_df = pd.DataFrame(results)
print("Initial Results:")
print(results_df[["wijknaam", "score", "pop_density"]])

# First check: Display neighborhoods based on initial satisfaction of properties
results_df = gpd.GeoDataFrame(results_df, geometry="geometry", crs=eindhoven_buurten.crs)
results_df = results_df.to_crs(epsg=4326)
initial_map = folium.Map(location=[51.44, 5.48], zoom_start=12, tiles="cartodbpositron")

for _, row in results_df.iterrows():
    # Add the "score" property to the GeoJSON feature
    geojson = folium.GeoJson(
        data={
            "type": "Feature",
            "geometry": mapping(row["geometry"]),
            "properties": {"score": row["score"]},  # Explicitly add the "score" property
        },
        style_function=lambda feature: {
            "fillColor": "green" if feature["properties"]["score"] == "pass" else "red",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.5,
        },
        tooltip=f"{row['wijknaam']} ({row['score']})"
    )
    geojson.add_to(initial_map)

    # Validate geometry before calculating centroid
    if row["geometry"].is_empty or not row["geometry"].is_valid:
        print(f"Skipping invalid or empty geometry for wijknaam: {row['wijknaam']}")
        continue

    # Calculate centroid
    centroid = row["geometry"].centroid

    # Add a visible caption at the centroid of the district
    folium.Marker(
        location=[centroid.y, centroid.x],
        icon=folium.DivIcon(
            html=f"""
                <div style="font-size: 10px; color: black; text-align: center; font-family: 'Times New Roman';">
                    {row['wijknaam']}
                </div>
            """
        )
    ).add_to(initial_map)

# Add a legend to the map - this is a custom HTML element, since Folium does not have
# native support for legend features.
legend_html = """
<div style="
    position: fixed; 
    bottom: 50px; left: 50px; width: 200px; height: 90px; 
    background-color: white; z-index: 1000; font-size: 14px; opacity: 0.75; 
    border: 2px solid black; padding: 10px; border-radius: 5px;">
    <b>Legend</b><br>
    <i style="background: green; width: 10px; height: 10px; display: inline-block; margin-right: 5px;"></i> Passing wijken<br>
    <i style="background: red; width: 10px; height: 10px; display: inline-block; margin-right: 5px;"></i> Failing wijken<br>
</div>
"""

legend_element = folium.Element(legend_html)
initial_map.get_root().html.add_child(legend_element)

# Title the map for clarity, using the density inputs:
if (minDensityInput < 3500):
    map_title = "Lowered-density Eindhoven 15-minute zoning"
else: 
    map_title = "Eindhoven 15-minute zoning using standardized population density"
    
title_html = f"""
    <div style="
        position: fixed; 
        top: 10px; left: 50%; transform: translateX(-50%);
        font-size: 10px; color: black; 
        z-index: 1000; text-align: center; background-color: white; 
        padding: 4px; border: 2px solid black; border-radius: 5px;">
        {map_title}
    </div>
"""
title_element = folium.Element(title_html)
initial_map.get_root().html.add_child(title_element)

# Save the initial map
initial_map.save("eindhoven_initial_check_map_with_legend.html")
print("Initial map with legend saved as eindhoven_initial_check_map_with_legend.html")

# Expand the zones that scored "low" to see if we can improve their scores by expanding the isochrone. 
# Note that this only tries for NEIGHBOURING zones.
def merge_neighborhoods(failing_neighborhoods, passing_neighborhoods):
    failing_neighborhoods = gpd.GeoDataFrame(failing_neighborhoods, geometry="geometry", crs=eindhoven_buurten.crs)
    passing_neighborhoods = gpd.GeoDataFrame(passing_neighborhoods, geometry="geometry", crs=eindhoven_buurten.crs)

    merged_results = []
    for fail in failing_neighborhoods.itertuples():
        neighbors = passing_neighborhoods[passing_neighborhoods.geometry.touches(fail.geometry)]
        merged = False
        for _, neighbor in neighbors.iterrows():
            merged_geometry = gpd.GeoSeries([fail.geometry, neighbor.geometry]).union_all()
            merged_isochrone = gpd.GeoSeries([fail.isochrone, neighbor.isochrone]).union_all()
            # Placeholder value, not used in final merging calculation.
            # pop_density = (fail.pop_density + neighbor.pop_density) / 2
            
            # Calculate actual population density based on area of merged geometry
            new_merged_area = merged_geometry.area
            new_population = (fail.pop_density * fail.geometry.area) + (neighbor.pop_density * neighbor.geometry.area)
            pop_density = new_population / new_merged_area if new_merged_area > 0 else 0
            if (
                POP_DENSITY_MIN <= pop_density <= POP_DENSITY_MAX
            ):
                merged_results.append({
                    "wijknaam": f"{fail.wijknaam} + {neighbor.wijknaam}",
                    "geometry": merged_geometry,
                    "isochrone": merged_isochrone,
                    "score": "pass",
                    "pop_density": pop_density,
                })
                print(f"Successfuly merged neighbourhoods {fail.wijknaam} and {neighbor.wijknaam}")
                merged = True
                break
        if not merged:
            print(f"Neighborhood {fail.wijknaam} could not be merged and will be eliminated.")
    return pd.DataFrame(merged_results)

failing_neighborhoods = results_df[results_df["score"] == "fail"]
passing_neighborhoods = results_df[results_df["score"] == "pass"]
merged_results = merge_neighborhoods(failing_neighborhoods, passing_neighborhoods)

# Ensure final_results is a GeoDataFrame
final_results = pd.concat([results_df[results_df["score"] == "pass"], merged_results])
final_results = gpd.GeoDataFrame(final_results, geometry="geometry", crs=eindhoven_buurten.crs)

# Debug: Check CRS and geometries
print(f"CRS of final_results before reprojecting: {final_results.crs}")
print(f"Number of empty geometries: {final_results.geometry.is_empty.sum()}")
print(f"Number of invalid geometries: {final_results.geometry.is_valid.sum()}")

# Fix invalid geometries
final_results["geometry"] = final_results.geometry.buffer(0)

# Drop rows with empty geometries
final_results = final_results[~final_results.geometry.is_empty]
print(final_results.geometry)

# Ensure CRS is set
if final_results.crs is None:
    final_results = final_results.set_crs(eindhoven_buurten.crs)

# Reproject to WGS84 (EPSG:4326)
final_results = final_results.to_crs(epsg=4326)

# Debug: Check the number of neighborhoods to plot
print(f"Number of neighborhoods to plot: {len(final_results)}")

# Debugging/Temporary solution: instead of trying to map this onto a folium map,
# I created a matplotlib visualization of the neighbourhoods we may consider as
# passing the tests.

# # Plot the final map
# m = folium.Map(location=[51.44, 5.48], zoom_start=12, tiles="cartodbpositron")
# for _, row in final_results.iterrows():
#     folium.GeoJson(
#         mapping(row["geometry"]),
#         style_function=lambda feature: {
#             "fillColor": "green" if row["score"] == "pass" else "red",
#             "color": "black",
#             "weight": 1,
#             "fillOpacity": 0.4,
#         },
#         tooltip=f"{row['wijknaam']} ({row['score']})"
#     ).add_to(m)

# m.save("eindhoven_final_15min_map.html")
# print("Final map saved as eindhoven_final_15min_map.html")