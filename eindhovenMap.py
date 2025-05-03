import osmnx as ox
import geopandas as gpd
import networkx as nx

import folium
import branca.colormap as cmap # a folium spin-off for colormapping.

### Quick description of what this does!
#### The goal of this ""model"" is to see how well the existing neighborhoods in Eindhoven
#### fit the criteria for a 15-minute city. It specifically uses OpenStreetMap's OSMnx 
#### library, which allows us to epxloit the existing neighborhood classifications of 
#### Eindhoven, and see which ones best fit the specified criteria (which can easily be)
#### changed by changing tags.

#### Ideally, the next step would be to not hard-code this but provide input on tags for
#### the 15-minute city, but that's for later. For now I also added "loading" statements,
#### because it takes a while to load.


# Configure criteria to roughly fit 15mC as defined by Carlos Moreno
# (so access to amenities via 15 minute walk... + bike in our case)
place = "Eindhoven, Netherlands"

# According to the Journal of Applied Physiology, average walking speed falls 
# between 4.0 - 5.9 km/h. Here, we use the lower bound, since the model is supposed
# to be as inclusive as possible.
average_walk_speed = 4

# On Copenhagen's website it says that an elderly person averages a 10 km/h
# cycling speed. For the same reason as stated above, we use this as our average.
average_bike_speed = 10

# It's a 15-minute city, so:
time_min = 15

walk_radius_m = average_walk_speed * 1000 / 60 * time_min    
bike_radius_m = average_bike_speed * 1000 / 60 * time_min    


# The "amenities" are lifted straight from the lsit of services I (Maciej) found during research 
# for the first part of the assignment. OSMnx uses a pretty odd manner of defining "leisure," 
# but in 

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

#  ---> Loading the walking/biking networks from OSMnx
print('Loading networks')

G_walk = ox.graph_from_place(place, network_type='walk')
G_bike = ox.graph_from_place(place, network_type='bike')

G_walk = ox.project_graph(G_walk)
G_bike = ox.project_graph(G_bike)

#  ---> Loading the amenities provided by tags
print('Loading amenities.')

gdf_amenities = ox.features_from_place(place, tags=tags)
gdf_amenities = gdf_amenities.to_crs(ox.graph_to_gdfs(G_walk, nodes=True, edges=False).crs)

# --> Set neighborhood boundaries.

# IMPORTANT: From what I understand, admin_level describes the key of the feature within
# the distribution hierarchy, alongside the boundary tag to define that we are using
# an administrative boundary. This feature is different per Nation, and for the Netherlands,
# admin_level = 10 will pertain to "woonplaatsen", so the non-autonomous municipal subdivisions,
# such as Centrum, Strijp, and Woensel (which is perfect for the primary elimination we want to achieve.) 

print('Loading boundaries.')

admin_tags = {"boundary": "administrative", "admin_level": "10"}
crs = G_walk.graph['crs']
neighborhoods = ox.features_from_place(place, tags=admin_tags)
neighborhoods = neighborhoods[neighborhoods.geometry.type.isin(['Polygon', 'MultiPolygon'])]
neighborhoods = neighborhoods.to_crs(crs)

# --> Perform rudimentary analysis.
results = []

# This analyzes access by walking/biking and compares the two as based on the entries in the
# previous tags category. I'll admit I got stuck here, so the first part of this is Copiloted.

def analyze_access(graph, center_node, radius_m, tags):
    # Get subgraph (this returns a subgraph around the node within the radius.)
    subgraph = nx.ego_graph(graph, center_node, radius=radius_m, distance='length')
    nodes, _ = ox.graph_to_gdfs(subgraph)
    isochrone_poly = nodes.geometry.unary_union.convex_hull
    inside = gdf_amenities[gdf_amenities.geometry.within(isochrone_poly)]

    # Set to hold categories that are covered by the 15-minute walk/bike zone
    categories_covered = set()

    # Check for matching amenities based on tags
    for k, v_list in tags.items():
        for v in v_list:
            match = inside[inside[k] == v]
            if not match.empty:
                categories_covered.add((k, v))
    return categories_covered

for idx, row in neighborhoods.iterrows():
    name = row.get('name', f"Neighborhood {idx}")
    centroid = row.geometry.centroid

    try:
        node_walk = ox.distance.nearest_nodes(G_walk, centroid.x, centroid.y)
        node_bike = ox.distance.nearest_nodes(G_bike, centroid.x, centroid.y)
    except:
        continue  # Skip invalid geometry


    walk_access = analyze_access(G_walk, node_walk, walk_radius_m, tags)
    try:
         bike_access = analyze_access(G_bike, node_bike, bike_radius_m, tags)
    except ValueError as e:
        print(f"Skipping bike access analysis: {e}")
        bike_access = set()  
        
    # To compare using a geodataframe, we create a dataset for each neighborhood from
    # the measurements we have defined.
    results.append({
        'name': name,
        'walk_score': len(walk_access),
        'bike_score': len(bike_access),
        'improvement': len(bike_access) - len(walk_access),
        'walk_amenities': walk_access,
        'bike_amenities': bike_access,
        'geometry': row.geometry
    })
    

print("Top neighborhoods by 15mC classification:")

# --> Results:

results_sorted = sorted(results, key=lambda x: -x['improvement'])

print("Top neighborhoods by 15mC classification:")
results_sorted = sorted(results, key=lambda x: -x['improvement'])

for r in results_sorted[:10]:
    print(f"Neighborhood: {r['name']} | Walk Score: {r['walk_score']} → Bike Score: {r['bike_score']} | Improvement: +{r['improvement']}")

gdf_results = gpd.GeoDataFrame(results, crs=neighborhoods.crs)


# Tried making a folium map: so far, just failure.
# min_improve = min(r['improvement'] for r in results)
# max_improve = max(r['improvement'] for r in results)
# colormap = cmap.LinearColormap(colors=['red', 'orange', 'yellow', 'green'],
#                              vmin=min_improve, vmax=max_improve,
#                              caption='Improvement (Bike - Walk)')

# # Assign a color to each neighborhood based on improvement
# gdf_results['color'] = gdf_results['improvement'].apply(colormap)

# # Create the Folium map
# center = neighborhoods.unary_union.centroid
# m = folium.Map(location=[center.y, center.x], zoom_start=12)

# ## Coloring the neighborhoods.
# for _, row in gdf_results.iterrows():
#     folium.GeoJson(
#         row['geometry'],
#         style_function=lambda feature, color=row['color']: {
#             'fillColor': color,
#             'color': 'black',
#             'weight': 1,
#             'fillOpacity': 0.6,
#         },
#         tooltip=folium.Tooltip(f"{row['name']}: +{row['improvement']} improvement"),
#     ).add_to(m)

# # Add color scale to the map
# colormap.add_to(m)


# m.save('eindhoven_15min_city_map.html')
# m