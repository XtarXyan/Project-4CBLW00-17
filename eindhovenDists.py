import osmnx as ox
import networkx as nx
import random

place = "Eindhoven, Netherlands"

print('Loading networks')

G_walk = ox.graph_from_place(place, network_type='walk')
G_bike = ox.graph_from_place(place, network_type='bike')

G_walk = ox.project_graph(G_walk)
G_bike = ox.project_graph(G_bike)

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
    ]
}

def calculate_distances(G, location_start, location_end=None):
    """
    Calculate the distance from a start location to all nodes in the graph.
    If an end location is provided, calculate the distance to that specific node.
    """
    # Get the nearest node to the start location
    start_node = ox.distance.nearest_nodes(G, location_start[1], location_start[0])
    
    if location_end:
        # Get the nearest node to the end location
        end_node = ox.distance.nearest_nodes(G, location_end[1], location_end[0])
        return nx.shortest_path_length(G, start_node, end_node, weight='length')
    
    # Calculate distances from the start node to all other nodes
    distances = nx.single_source_dijkstra_path_length(G, start_node, cutoff=None, weight='length')
    
    return distances

def get_nearest_amenities(G, location, tag, max_count):
    """
    Get the nearest max_count amenities of a specific tag type from a given location.
    """
    # Get the nearest node to the location
    start_node = ox.distance.nearest_nodes(G, location[1], location[0])
    
    # Get the nodes of the amenities of the specified tag
    amenity_nodes = [node for node, data in G.nodes(data=True) if data.get('amenity') in tags[tag]]
    
    # Calculate distances to all amenity nodes
    distances = {node: nx.shortest_path_length(G, start_node, node, weight='length') for node in amenity_nodes}
    
    # Sort by distance and get the nearest amenities
    nearest_amenities = sorted(distances.items(), key=lambda x: x[1])[:max_count]
    
    return nearest_amenities
   
def get_nearest_amenities_inbetween(G, location_start, location_end, tag, max_count):
    """
    Get the nearest max_count amenities of a specific tag type between two locations.
    """
    # Get the nearest nodes to the start and end locations
    start_node = ox.distance.nearest_nodes(G, location_start[1], location_start[0])
    end_node = ox.distance.nearest_nodes(G, location_end[1], location_end[0])
    
    # Get the nodes of the amenities of the specified tag
    amenity_nodes = [node for node, data in G.nodes(data=True) if data.get('amenity') in tags[tag]]
    
    # Calculate distances to all amenity nodes
    distances = {node: nx.shortest_path_length(G, start_node, node, weight='length') + nx.shortest_path_length(G, node, end_node, weight='length') for node in amenity_nodes}
    
    # Sort by distance and get the nearest amenities
    nearest_amenities = sorted(distances.items(), key=lambda x: x[1])[:max_count]
    
    return nearest_amenities