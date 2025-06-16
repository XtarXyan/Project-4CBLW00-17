import osmnx as ox
import networkx as nx
from shapely import unary_union
from sklearn.neighbors import BallTree
import geopandas as gpd
import pandas as pd
import numpy as np
import json

# For pickling things that need to load faster
import pickle
import os

# For profiling the code
from line_profiler import profile, LineProfiler
import atexit

profile = LineProfiler()

# Register the profiler to save the results on exit
atexit.register(profile.print_stats)


# BallTree approach

# This script loads the networks and POIs from OSM for a specific place
# and creates a BallTree for fast nearest neighbor queries.
# There are two networks: one for walking and one for biking. There are also two types of calculations that this script can do:
# 1. Get the nearest amenities of a specific type from a given location.
# 2. Get the nearest amenities of a specific type in between two locations.

# Particularly the second one is a little complicated. Essentially it computes an "optimization"
# that a person might be looking for by stopping along the way between two locations.
# The script also caches the results in pickle files for faster loading in the future.

# Two other types of caching happen at runtime:
# 1. The BallTree for each tag list is cached in memory.
# 2. The amenity nodes are cached in memory to avoid recomputing them multiple times.
# Surprisingly the second one is quite expensive which is why it is cached.

tags = {
    'building': ['apartments', 'house'],  # Residences
    'office': ['yes'],  # All workplaces, since OSM has an incredible amount of identifiers.
    'amenity': [
        'pharmacy', 'hospital', 'clinic', 'doctors', 
        'school', 'kindergarten', 'college', 'university',  
        'cafe', 'restaurant', 'bar', 'cinema', 'theatre',  
        'community_centre', 'library', 'bicycle_rental',
        'place_of_worship',
        # Place of worship can be considered "key" ammenity for some, so should be included.
        'locker', 'parcel_locker', 'public_bathroom', 'public_toilet',
    ],
    'leisure': [
        'park', 'fitness_centre', 'sports_centre', 'stadium', 
        'dog_park', 'pitch', 'swimming_pool', 'playground',
        'nature_reserve'
    ],
    'shop' : [
        'supermarket', 'convenience', 'bakery', 'greengrocer', 'butcher',
        'department_store', 'general', 'cosmetics', 'stationery'
    ]
}

features = gpd.GeoDataFrame()
amenities = gpd.GeoDataFrame()
leisure = gpd.GeoDataFrame()
shops = gpd.GeoDataFrame()
offices = gpd.GeoDataFrame()
residences = gpd.GeoDataFrame()
universities = gpd.GeoDataFrame()
workplaces = gpd.GeoDataFrame()
geodata = gpd.GeoSeries()

data = None

G_fp = "graph.pickle"
geodata_fp = "geodata.pickle"
features_fp = "features.pickle"

all_pairs_distances = None
all_pairs_distances_fp = "all_pairs_distances.feather"

def initialize():
    """
    Initialize the script by loading the necessary data.
    This function should be called before any other function.
    """

    global data, all_pairs_distances
    print("Initializing data...")

    data = get_data()
    if data is None:
        raise ValueError("Data for the specified place could not be loaded.")
    all_pairs_distances = get_all_pairs_distances(data["graph"])
    return data, all_pairs_distances

def get_data():

    # Check if the data is already loaded
    global data
    if data is not None:
        print("Data already loaded.")
        return data

    global features, amenities, leisure, shops, offices, universities, residences, workplaces
    global G_fp, geodata_fp, features_fp

    graph = None
    geodata = None

    with open("refinedMap.geojson") as openedFile:
        polygonGeoJSON = json.load(openedFile)
    if not polygonGeoJSON:
        raise ValueError("GeoJSON file is empty or not found.")

    # Check if the geodata file exists
    if os.path.exists(geodata_fp):
        print(f"Loading geodata from {geodata_fp}...")
        with open(geodata_fp, 'rb') as f:
            geodata = pickle.load(f)
    else:
        print(f"Geodata file {geodata_fp} does not exist. Creating new geodata.")
        # Convert GeoJSON to GeoDataFrame
        geoDataEindhoven = gpd.GeoDataFrame.from_features(polygonGeoJSON["features"])
        # Ensure the GeoDataFrame has the correct CRS
        geoDataEindhoven.crs = "EPSG:4326"
        geodata = unary_union(geoDataEindhoven.geometry)

        # Save the geodata to a pickle file
        with open(geodata_fp, 'wb') as f:
            print(f"Saving geodata to {geodata_fp}...")
            pickle.dump(geodata, f)

    # Check if the graph file exists
    if os.path.exists(G_fp):
        print(f"Loading graph from {G_fp}...")
        with open(G_fp, 'rb') as f:
            graph = pickle.load(f)
    else:
        print(f"Graph file {G_fp} does not exist. Creating new graph.")
        # Get the graph of the area
        graph = ox.graph_from_polygon(geodata, network_type="bike", simplify=True)
        # Ensure the graph is undirected
        graph = graph.to_undirected()
        # Save the graph to a pickle file
        with open(G_fp, 'wb') as f:
            print(f"Saving graph to {G_fp}...")
            pickle.dump(graph, f)
        

    # Convert the graph to a GeoDataFrame
    nodes, edges = ox.graph_to_gdfs(graph)
    # Ensure the nodes and edges have the correct CRS
    nodes.crs = "EPSG:4326"
    edges.crs = "EPSG:4326"

    features = initialize_features(graph, geodata)
    amenities, leisure, shops, offices, universities, residences = initialize_subcategories(features)
    workplaces = gpd.GeoDataFrame(pd.concat([amenities, offices, shops], ignore_index=True))
    workplaces.crs = "EPSG:4326"

    return {
        "graph": graph,
        "nodes": nodes,
        "edges": edges,
        "geodata": geodata,
        "features": features,
        "amenities": amenities,
        "leisure": leisure,
        "shops": shops,
        "offices": offices,
        "universities": universities,
        "residences": residences,
        "workplaces": workplaces
        }



def initialize_features(graph, geodata):

    global features
    global features_fp

    # Check if the features file exists
    if os.path.exists(features_fp):
        print(f"Loading features from {features_fp}...")
        # Load the features from the pickle file
        with open(features_fp, 'rb') as f:
            features = pickle.load(f)
    else:
        print(f"Features file {features_fp} does not exist. Creating new features.")
        
        # Print the type of the polygon
        print(f"Polygon type: {type(geodata)}")
        
        # Handle MultiPolygon by iterating over each polygon
        if geodata.geom_type == "MultiPolygon":
            features_list = []
            for poly in geodata.geoms:
                features_list.append(ox.features_from_polygon(poly, tags=tags))
            features = pd.concat(features_list, ignore_index=True)
        else:
            features = ox.features_from_polygon(geodata, tags=tags)
        # Convert to GeoDataFrame
        features = gpd.GeoDataFrame(features, geometry='geometry', crs="EPSG:4326")
        # Save the features to pickle file
        print(f"Saving features to {features_fp}...")
        with open(features_fp, 'wb') as f:
            pickle.dump(features, f)
        

    # Ensure the features have a 'main_tag' column 
    if 'main_tag' not in features.columns:
        features['main_tag'] = features['amenity'].fillna(features['office']).fillna(features['shop']).fillna(features['leisure']).fillna(features['building'])
    # If no main_tag found, set it to 'unknown'
    if features['main_tag'].isnull().all():
        features['main_tag'] = 'unknown'
    # Add centroid column
    features["centroid"] = features.geometry.centroid

    # Extract coordinates from centroids
    x = features["centroid"].x
    y = features["centroid"].y

    # Get nearest nodes for each feature in batch
    features["nearest_node"] = ox.nearest_nodes(
        graph, X=x, Y=y, return_dist=False)

    invalid_nodes = ~features["nearest_node"].isin(graph.nodes)
    if invalid_nodes.any():
        print(f"Warning: {invalid_nodes.sum()} features have nearest_node not in the graph!")

    # Ensure the features have a 'geometry' column
    if 'geometry' not in features.columns:
        raise ValueError("Features GeoDataFrame does not contain a 'geometry' column.")
    
    return features
    

def initialize_subcategories(features):
    """
    Initialize the amenities GeoDataFrame from the features.
    """
    global amenities
    global leisure
    global shops
    global offices
    global universities
    global residences
    global tags
    
    # Filter for amenities, leisure, and shops
    amenities = features[features['main_tag'].isin(tags['amenity'])]
    leisure = features[features['main_tag'].isin(tags['leisure'])]
    shops = features[features['main_tag'].isin(tags['shop'])]
    offices = features[features['main_tag'].isin(tags['office'])]
    universities = features[features['main_tag'].isin(['university', 'college'])]
    residences = features[features['main_tag'].isin(tags['building'])]
    
    # Add centroid column
    amenities["centroid"] = amenities.geometry.centroid
    leisure["centroid"] = leisure.geometry.centroid
    shops["centroid"] = shops.geometry.centroid
    offices["centroid"] = offices.geometry.centroid
    universities["centroid"] = universities.geometry.centroid
    residences["centroid"] = residences.geometry.centroid
    
    # Convert to GeoDataFrame
    amenities = gpd.GeoDataFrame(amenities, geometry='geometry', crs="EPSG:4326")
    leisure = gpd.GeoDataFrame(leisure, geometry='geometry', crs="EPSG:4326")
    shops = gpd.GeoDataFrame(shops, geometry='geometry', crs="EPSG:4326")
    offices = gpd.GeoDataFrame(offices, geometry='geometry', crs="EPSG:4326")
    universities = gpd.GeoDataFrame(universities, geometry='geometry', crs="EPSG:4326")
    residences = gpd.GeoDataFrame(residences, geometry='geometry', crs="EPSG:4326")

    # Ensure the amenities have a 'main_tag' column
    if 'main_tag' not in amenities.columns:
        amenities['main_tag'] = amenities['amenity']

    if 'main_tag' not in leisure.columns:
        leisure['main_tag'] = leisure['leisure']

    if 'main_tag' not in shops.columns:
        shops['main_tag'] = shops['shop']

    if 'main_tag' not in offices.columns:
        offices['main_tag'] = offices['office']

    if 'main_tag' not in universities.columns:
        universities['main_tag'] = universities['amenity']

    if 'main_tag' not in residences.columns:
        residences['main_tag'] = residences['building']
    
    return amenities, leisure, shops, offices, universities, residences
    

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


tags_main = [
    'apartments', 'house', 'office', 'university', 'college',
    'pharmacy', 'hospital', 'clinic', 'doctors', 
    'school', 'kindergarten', 'college', 'university',  
    'cafe', 'restaurant', 'bar', 'cinema', 'theatre',  
    'community_centre', 'library', 'bicycle_rental',
    'place_of_worship', 
    'park', 'playground', 'fitness_centre', 'sports_centre', 'stadium', 
    'dog_park', 'pitch', 'swimming_pool', 'nature_reserve',
    'supermarket', 'convenience', 'bakery', 'greengrocer', 'butcher',
    'department_store', 'general', 'cosmetics', 'stationery',
    'hub', # For mobility
    'locker', 'parcel_locker', 'public_bathroom', 'public_toilet', # For public facilities that may be in hubs
]

def get_all_pairs_distances(G):
    """
    Calculate all pairs shortest path distances for the given graph and nodes.
    """
    global all_pairs_distances
    if all_pairs_distances is not None:
        return all_pairs_distances
    
    df = pd.DataFrame()
    
    global all_pairs_distances_fp
    # Check if the all pairs distances file exists
    if os.path.exists(all_pairs_distances_fp):
        print(f"Loading all pairs distances from {all_pairs_distances_fp}...")
        # Load the distances from the feather file
        df = pd.read_feather(all_pairs_distances_fp)
        
    else:
        print(f"All pairs distances file {all_pairs_distances_fp} does not exist. Calculating new distances.")
        # Use Floyd-Warshall algorithm to calculate all pairs shortest path distances
        all_pairs_distances = dict(nx.all_pairs_dijkstra_path_length(G, weight='length'))
        
        # Convert to DataFrame
        df = pd.DataFrame(all_pairs_distances).T
        df.index.name = 'from_node'
        df.columns.name = 'to_node'
        # Save the all pairs distances to a feather file
        print(f"Saving all pairs distances to {all_pairs_distances_fp}...")
        df.to_feather(all_pairs_distances_fp)
    
    return df

def get_distances(G, start_node, end_node=None):
    # Get the shortest path length between two nodes in the graph based on the all pairs distances
    global all_pairs_distances
    if end_node is not None:
        try:
            return all_pairs_distances.loc[start_node, end_node]
        except KeyError:
            # Compute and update if not present
            shortest_path = nx.shortest_path_length(G, start_node, end_node, weight='length')
            all_pairs_distances.loc[start_node, end_node] = shortest_path
            all_pairs_distances.loc[end_node, start_node] = shortest_path
            return shortest_path
    else:
        try:
            return all_pairs_distances.loc[start_node]
        except KeyError:
            raise KeyError(f"Start node {start_node} not in all_pairs_distances DataFrame.")

# Create cache for (key, value) → BallTree
feature_trees = {}
amenity_node_cache = {}

def get_tree_for_tags(tag_list):
    """
    Get a BallTree for a specific list of tag values (e.g., 'pharmacy').
    """
    tag_list = tuple(sorted(tag_list))
    # Check if the tree is already cached
    if tag_list in feature_trees:
        return feature_trees[tag_list]
    
    # Filter POIs by the tag values
    filtered = features[features['main_tag'].isin(tag_list)]
    if filtered.empty:
        raise ValueError(f"No POIs found for tag list '{tag_list}'")
    
    print("Number of amenities of type", tag_list, ":", len(filtered))
    
    # Convert coordinates to radians for BallTree
    coords = np.radians(np.column_stack((filtered["centroid"].y.values, filtered["centroid"].x.values)))
    tree = BallTree(coords, metric="haversine")
    print(f"Created BallTree for tags: {tag_list}")
    feature_trees[tag_list] = tree
    return tree

def get_cached_amenity_node(G, key):
    if key in amenity_node_cache:
        return amenity_node_cache[key]
    node = ox.distance.nearest_nodes(G, X=key[0], Y=key[1])
    amenity_node_cache[key] = node
    return node
        
# @profile
def get_nearest_amenities(G, location, tag_list, max_count):
    """
    Get the nearest max_count amenities of specific tag types from a given location.
    """

    # Ensure there are more or equal amenities than max_count
    print("Getting nearest amenities from location")
    if len(features[features['main_tag'].isin(tag_list)]) < max_count:
        max_count = len(features[features['main_tag'].isin(tag_list)])

    if not isinstance(location, tuple) or len(location) != 2:
        raise ValueError("Location must be a tuple of (longitude, latitude).")
    
    print("Getting cached amenity node")
    location_node = get_cached_amenity_node(G, location)
    
    # Ensure the location node is valid
    if location_node not in G.nodes:
        raise ValueError(f"Location node {location_node} is not a valid node in the graph.")
    
    print("Getting nearest amenities from node")
    return get_nearest_amenities_from_node(G, location_node, tag_list, max_count)

# @profile
def get_nearest_amenities_from_node(G, node, tag_list, max_count):
    """
    Get the nearest max_count amenities of specific tag types from a given node.
    """
    # Ensure there are more or equal amenities than max_count
    filtered = features[features['main_tag'].isin(tag_list)]
    if len(filtered) < max_count:
        max_count = len(filtered)
    if filtered.empty:
        raise ValueError(f"No POIs found for tags: {tag_list}")

    coords = (G.nodes[node]['y'], G.nodes[node]['x'])  # (lat, lon) for BallTree
    tree = get_tree_for_tags(tag_list)
    dist, ind = tree.query([np.radians(coords)], k=max_count)
    nearest_amenities = filtered.iloc[ind[0]]

    # Vectorized assignment of distances
    location_distances = get_distances(G, node)

    # Check if location_distances has NaN values
    if any(pd.isna(location_distances.values)):
        raise ValueError("Location distances contain NaN values, which may indicate an issue with the graph or node.")

    nearest_amenities = nearest_amenities.assign(
        distance=nearest_amenities['nearest_node'].map(location_distances)
    )

    
    return nearest_amenities.sort_values("distance")

# @profile
def get_nearest_amenities_inbetween(G, start_node, end_node, tag_list, max_count, path_resolution=5):
    """
    Get the nearest max_count amenities of specific tag types in between two locations.
    """

    # Validate inputs
    if not isinstance(start_node, (int, np.integer)) or not isinstance(end_node, (int, np.integer)):
        raise ValueError(f"Start and end nodes must be integers representing node IDs. Types are {type(start_node)} and {type(end_node)}.")
    if not isinstance(tag_list, tuple) or not all(isinstance(tag, str) for tag in tag_list):
        raise ValueError("Tag list must be a tuple of strings representing tag types.")
    if not isinstance(max_count, int) or max_count <= 0:
        raise ValueError("Max count must be a positive integer.")
    
    # Ensure the start and end nodes are in the graph
    if start_node not in G.nodes or end_node not in G.nodes:
        raise ValueError(f"Start and end nodes must be valid nodes in the graph. Start node: {start_node}, End node: {end_node}")
    if start_node == end_node:
        # If the start and end nodes are the same, return nearest amenities from that node
        return get_nearest_amenities_from_node(G, start_node, tag_list, max_count)

    # Ensure there are more or equal amenities than max_count
    if len(features[features['main_tag'].isin(tag_list)]) < max_count:
        max_count = len(features[features['main_tag'].isin(tag_list)])

    # Get the shortest path between the two nodes
    path = nx.shortest_path(G, start_node, end_node, weight='length')
    radius_m = get_distances(G, start_node, end_node) / path_resolution
    if radius_m <= 0:
        raise ValueError("The distance between the start and end nodes is zero or negative, cannot sample nodes along the path.")

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

    # Get the coordinates of the path
    path_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in sampled_nodes]
    if not path_coords:
        raise ValueError("No valid path found between the start and end locations.")

    # Filter POIs by the list of tags
    filtered = features[features['main_tag'].isin(tag_list)].copy()
    if filtered.empty:
        raise ValueError(f"No POIs found for tags: {tag_list}")

    tree = get_tree_for_tags(tag_list)

    # Query the BallTree for the k nearest amenities from each sampled node
    k = max_count  # or a bit more, to ensure enough unique amenities
    results = []
    indices = []
    for coord in path_coords:
        coord_rad = np.radians([coord])
        dist, ind = tree.query(coord_rad, k=k)
        for d, idx in zip(dist[0], ind[0]):
            amenity_distance = d * 6371000
            row = filtered.iloc[int(idx)].copy()
            row["distance"] = round(amenity_distance, 2)
            results.append(row)
            indices.append(filtered.index[int(idx)])  # Preserve original index

    gdf = gpd.GeoDataFrame(results)
    gdf.index = indices  # Set the index to the original features index

    if not gdf.empty:
        # Use get_distances for fast lookup
        start_lengths = get_distances(G, start_node)
        end_lengths = get_distances(G, end_node)
        gdf["distance"] = gdf["nearest_node"].apply(
            lambda n: start_lengths.get(n, np.inf) + end_lengths.get(n, np.inf)
        )
        gdf = gdf.sort_values("distance").head(max_count)
    else:
        gdf = gpd.GeoDataFrame(columns=list(filtered.columns) + ["distance"])
    return gdf

# Example usage
@profile
def example_usage():
    # Initialize the data
    data, all_pairs_distances = initialize()
    
    # Example coordinates for testing
    start_coords = (5.4697, 51.4416)  # Eindhoven coordinates
    start_node = ox.distance.nearest_nodes(data["graph"], X=start_coords[0], Y=start_coords[1])
    tag_list = ['pharmacy', 'cafe', 'restaurant']
    
    # Get nearest amenities from a specific location
    nearest_amenities = get_nearest_amenities(data["graph"], start_coords, tag_list, max_count=5)
    print(nearest_amenities)

    # Get node from coordinates
    node = ox.distance.nearest_nodes(data["graph"], X=start_coords[0], Y=start_coords[1])
    print(f"Nearest node to {start_coords} is {node}.")
    # Get nearest amenities from a node
    nearest_amenities_from_node = get_nearest_amenities_from_node(data["graph"], node, tag_list, max_count=5)
    print(nearest_amenities_from_node)
    
    # Get nearest amenities in between two locations
    end_coords = (5.4700, 51.4420)  # Another point in Eindhoven
    end_node = ox.distance.nearest_nodes(data["graph"], X=end_coords[0], Y=end_coords[1])
    nearest_inbetween = get_nearest_amenities_inbetween(data["graph"], start_node, end_node, tag_list, max_count=5)
    print(nearest_inbetween)

# example_usage()