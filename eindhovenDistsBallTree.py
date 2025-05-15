import osmnx as ox
import networkx as nx
from sklearn.neighbors import BallTree
import geopandas as gpd
import pandas as pd
import numpy as np

# For pickling things that need to load faster
import pickle
import os

# BallTree approach

place = 'Eindhoven, Netherlands'

print('Loading networks v4')

# Load the pickle filepaths if they exist
G_walk_fp = "G_walk.gml"
G_bike_fp = "G_bike.gml"
# Check if the files exist
if os.path.exists(G_walk_fp) and os.path.exists(G_bike_fp):
    print("Loading networks from files...")
    G_walk = ox.load_graphml(G_walk_fp)
    G_bike = ox.load_graphml(G_bike_fp)
else:
    # Load the networks
    print("Loading networks from OSM. This will take a while.")
    G_walk = ox.graph_from_place(place, network_type='walk', retain_all=False)
    G_bike = ox.graph_from_place(place, network_type='drive', retain_all=False)
    G_bike = G_bike.to_undirected()

    # Save the networks to files
    ox.save_graphml(G_walk, filepath=G_walk_fp)
    ox.save_graphml(G_bike, filepath=G_bike_fp)

tags = {
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
    ]
}

tags_amenities = {
     'amenity': [
        'pharmacy', 'hospital', 'clinic', 'doctors', 
        'school', 'kindergarten', 'college', 'university',  
        'cafe', 'restaurant', 'bar', 'cinema', 'theatre',  
        'community_centre', 'library', 'bicycle_rental',
        'place_of_worship', 
    ]
}


def load_from_pickle(fp):
    """
    Load a GeoDataFrame from a pickle file.
    """
    data = None
    if os.path.exists(fp):
        print(f"Loading data from {fp}...")
        with open(fp, 'rb') as f:
            data = pickle.load(f)
    return data

def save_to_pickle(data, fp):
    """
    Save a GeoDataFrame to a pickle file.
    """
    print(f"Saving data to {fp}...")
    with open(fp, 'wb') as f:
        pickle.dump(data, f)


amenities_fp = "amenities.pickle"
offices_fp = "offices.pickle"
residences_fp = "residences.pickle"
pois_fp = "pois.pickle"

# Check if the amenities file exists
amenities = load_from_pickle(amenities_fp)
if amenities is None:
    print("Loading amenities from OSM.")
    # Get the amenities of the area
    amenities = ox.features_from_place(place, tags=tags_amenities)
    # Filter for valid geometry types
    amenities = amenities[amenities.geometry.type.isin(["Polygon", "MultiPolygon", "Point"])]
    # Add centroid column
    amenities["centroid"] = amenities.geometry.centroid
    # Convert to GeoDataFrame
    amenities = gpd.GeoDataFrame(amenities, geometry='geometry', crs="EPSG:4326")
    # Save the amenities to a pickle file
    save_to_pickle(amenities, amenities_fp)

# Get the offices of the area
offices = load_from_pickle(offices_fp)
if offices is None:
    print("Loading offices from OSM.")
    # Get the offices of the area
    offices = ox.features_from_place(place, tags={'office': True})
    # Filter for valid geometry types
    offices = offices[offices.geometry.type.isin(["Polygon", "MultiPolygon", "Point"])]
    # Add centroid column
    offices["centroid"] = offices.geometry.centroid
    # Convert to GeoDataFrame
    offices = gpd.GeoDataFrame(offices, geometry='geometry', crs="EPSG:4326")
    # Save the offices to a pickle file
    save_to_pickle(offices, offices_fp)
    
# Combine the amenities and offices features
workplaces = gpd.GeoDataFrame(pd.concat([amenities, offices], ignore_index=True))

# Get the residences of the area
residences = load_from_pickle(residences_fp)
if residences is None:
    print("Loading residences from OSM. This will take a LONG while.")
    # Get the residences of the area
    residences = ox.features_from_place(place, tags={'building': ['apartments', 'house']})
    # Convert to GeoDataFrame
    residences = gpd.GeoDataFrame(residences, geometry='geometry', crs="EPSG:4326")
    # Save the residences to a pickle file
    save_to_pickle(residences, residences_fp)


# Get the POIs of the area
pois = load_from_pickle(pois_fp)
if pois is None:
    print("Loading POIs from OSM.")
    # Get the POIs of the area
    pois = ox.features_from_place(place, tags=tags)
    # Filter for valid geometry types
    pois = pois[pois.geometry.type.isin(["Polygon", "MultiPolygon", "Point"])]
    # Add centroid column
    pois["centroid"] = pois.geometry.centroid
    # Convert to GeoDataFrame with centroid as geometry
    pois = gpd.GeoDataFrame(pois, geometry="centroid", crs="EPSG:4326")
    # Combine amenity and leisure tags
    pois["main_tag"] = pois["amenity"].combine_first(pois["leisure"])
    # Save the POIs to a pickle file
    save_to_pickle(pois, pois_fp)

tags = [
    'pharmacy', 'hospital', 'clinic', 'doctors', 
    'cafe', 'restaurant', 'bar', 'cinema', 'theatre',  
    'community_centre', 'library', 'bicycle_rental',
    'place_of_worship', 
    'park', 'fitness_centre', 'sports_centre', 'stadium', 
    'dog_park', 'pitch', 'swimming_pool'
]

print(pois.head())

print(pois['main_tag'].unique())

print(len(pois), "POIs loaded")


# Create cache for (key, value) → BallTree
poi_trees = {}
amenity_node_cache = {}

def get_tree_for_tags(tag_list):
    """
    Get a BallTree for a specific list of tag values (e.g., 'pharmacy').
    """
    tag_list = tuple(sorted(tag_list))
    # Check if the tree is already cached
    if tag_list in poi_trees:
        return poi_trees[tag_list]
    
    # Filter POIs by the tag values
    filtered = pois[pois['main_tag'].isin(tag_list)].copy()

    if filtered.empty:
        raise ValueError(f"No POIs found for tag list '{tag_list}'")
    
    # Convert coordinates to radians for BallTree
    coords = np.radians(filtered.geometry.apply(lambda p: (p.y, p.x)).tolist())
    tree = BallTree(coords, metric="haversine")
    print(f"Created BallTree for tags: {tag_list}")
    poi_trees[tag_list] = tree
    return tree
    


def calculate_distances(G, location_start, location_end):
    """
    Calculate the distance from a start location to all nodes in the graph.
    """
    # Get the nearest node to the start location
    start_node = ox.distance.nearest_nodes(G, X=location_start[0], Y=location_start[1])
    
    # Get the nearest node to the end location
    end_node = ox.distance.nearest_nodes(G, X=location_end[0], Y=location_end[1])
    print(f"Start node: {start_node}, End node: {end_node}")
    return nx.shortest_path_length(G, start_node, end_node, weight='length')
        
    


def get_nearest_amenities(G, location, tag_list, max_count):
    """
    Get the nearest max_count amenities of specific tag types from a given location.
    """
    lat = location[1]
    lon = location[0]
    coords = np.radians((lat, lon))
    location_node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
    print(coords)
    
    # Filter POIs by the list of tags
    filtered = pois[pois['main_tag'].isin(tag_list)].copy()
    if filtered.empty:
        raise ValueError(f"No POIs found for tags: {tag_list}")
    
    tree = get_tree_for_tags(tag_list)
    
    # Query the BallTree for the nearest amenities
    dist, ind = tree.query([coords], k=max_count)
    nearest_amenities = filtered.iloc[ind[0]].copy()
    for amenity in nearest_amenities.itertuples():
        amenity_node = ox.distance.nearest_nodes(G, X=amenity.geometry.centroid.x, Y=amenity.geometry.centroid.y)
        amenity_distance = nx.shortest_path_length(G, location_node, amenity_node, weight='length')
        nearest_amenities.at[amenity.Index, 'distance'] = amenity_distance

    return nearest_amenities.sort_values("distance")

    
   
def get_nearest_amenities_inbetween(G, location_start, location_end, tag_list, max_count, path_resolution=5):
    """
    Get the nearest max_count amenities of specific tag types in between two locations.
    """
    # Get the nearest node to the start and end locations
    start_node = ox.distance.nearest_nodes(G, X=location_start[0], Y=location_start[1])
    end_node = ox.distance.nearest_nodes(G, X=location_end[0], Y=location_end[1])

    print("1")
    
    # Get the shortest path between the two nodes
    path = nx.shortest_path(G, start_node, end_node, weight='length')
    radius_m = nx.shortest_path_length(G, start_node, end_node, weight='length') / path_resolution

    print("2")

    # Sample nodes along the path at intervals of radius_m
    sampled_nodes = [path[0]]
    accumulated = 0
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v)
        # If multiple edges, take the first one
        if isinstance(edge_data, dict):
            edge_length = edge_data[list(edge_data.keys())[0]].get('length', 0)
        else:
            edge_length = edge_data.get('length', 0)
        accumulated += edge_length
        if accumulated >= radius_m:
            sampled_nodes.append(v)
            accumulated = 0
    if sampled_nodes[-1] != path[-1]:
        sampled_nodes.append(path[-1])

    print("3")
    
    # Get the coordinates of the path
    path_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in sampled_nodes]
    if not path_coords:
        raise ValueError("No valid path found between the start and end locations.")
    
    print("4")
    
    # Filter POIs by the list of tags
    filtered = pois[pois['main_tag'].isin(tag_list)].copy()
    if filtered.empty:
        raise ValueError(f"No POIs found for tags: {tag_list}")
    
    print("5")
    
    tree = get_tree_for_tags(tag_list)
    
    # Query the BallTree for the k nearest amenities from each sampled node
    k = max_count  # or a bit more, to ensure enough unique amenities
    results = []
    for coord in path_coords:
        coord_rad = np.radians([coord])
        dist, ind = tree.query(coord_rad, k=k)
        for d, idx in zip(dist[0], ind[0]):
            amenity_distance = d * 6371000
            row = filtered.iloc[int(idx)].copy()
            row["distance"] = round(amenity_distance, 2)
            results.append(row)

    print("6")

    gdf = gpd.GeoDataFrame(results)
    if "osmid" in gdf.columns:
        gdf = gdf.drop_duplicates(subset=["osmid"])
    else:
        gdf = gdf.drop_duplicates(subset=["geometry"])
    # ...after deduplication and before calculating distances...
    if not gdf.empty:
        def get_cached_amenity_node(geometry):
            # Use WKT as a unique key for geometry; or use amenity ID if available
            key = geometry.wkt
            if key in amenity_node_cache:
                return amenity_node_cache[key]
            node = ox.distance.nearest_nodes(G, X=geometry.centroid.x, Y=geometry.centroid.y)
            amenity_node_cache[key] = node
            return node

        gdf["amenity_node"] = gdf.geometry.apply(get_cached_amenity_node)
        amenity_nodes = gdf["amenity_node"].unique()
        # Batch shortest path lengths
        start_lengths = nx.single_source_dijkstra_path_length(G, start_node, weight='length')
        end_lengths = nx.single_source_dijkstra_path_length(G, end_node, weight='length')
        gdf["distance"] = gdf["amenity_node"].apply(
            lambda n: start_lengths.get(n, np.inf) + end_lengths.get(n, np.inf)
        )
        gdf = gdf.drop(columns=["amenity_node"])
        gdf = gdf.sort_values("distance").head(max_count)
    else:
        gdf = gpd.GeoDataFrame(columns=list(filtered.columns) + ["distance"])
    return gdf

#print(get_tree_for_tags(['pharmacy', 'park']))
#print(get_nearest_amenities(G_bike, (5.486, 51.449), ['pharmacy', 'park'], 5))
print(get_nearest_amenities_inbetween(G_walk, (5.486, 51.449), (5.485, 51.448), ['pharmacy', 'park'], 5))