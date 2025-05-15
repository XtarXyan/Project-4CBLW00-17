# -*- coding: utf-8 -*-
import eindhovenDistsBallTree as dists
import pandas as pd
import random


# Agent class simulates the behavior of an agent in a city.
# The agent has a residence and work location, and can travel to amenities based on their preferences and travel modes.
# The agent's travel behavior is influenced by their "curiosity" and the frequency of visiting work and amenities.
# The agent can travel by walking or biking, and the travel time is calculated based on the distance to the destination and the speed of the chosen mode of transport.
# The agent's daily activities are simulated, including travel to work and visits to amenities.
# The total travel time for the day is calculated and returned.
# Thanks Copilot for writing my comments

class person:
    def __init__(self, residence_coords, work_coords=None, walk_speed=4, bike_speed=10, bike_freq=0.0, work_freq=1.0, amenity_freqs=None, curiosity=1.0, seed=0):
        self.residence_coords = residence_coords  # This should be a tuple (latitude, longitude)
        self.work_coords = work_coords
        self.work_freq = work_freq  # Frequency of going to work (1.0 means every day)
        self.walk_speed = walk_speed  
        self.bike_speed = bike_speed
        self.bike_freq = bike_freq # Frequency of biking (0.0 means never, 1.0 means always)
        self.amenity_freqs = amenity_freqs # Dictionary of amenity type lists and their frequencies (0.0 to 1.0)
        self.curiosity = curiosity # Chance the agent will (recursively) pick the nearest amenity of a certain type over the following nearest one.
        self.work_travel_time = 0
        random.seed(seed)
        self.df_travel = pd.DataFrame(columns=["travel_time", "mode", "amenity_type", "amenity_coords"])
        self.initialize_distances()

    def travel_time(self, distance, mode='walk'):
        if mode == 'walk':
            speed = self.walk_speed
        elif mode == 'bike':
            speed = self.bike_speed
        else:
            raise ValueError("Mode must be 'walk' or 'bike'")
        
        time_hours = (distance / 1000) / speed
        return time_hours * 60
    
    def initialize_distances(self):
        G_walk = dists.G_walk
        G_bike = dists.G_bike

        # Calculate distances from residence to all nodes in the graph
        """
        self.distances_residence = {
            'walk': dists.calculate_distances(G_walk, self.residence_coords),
            'bike': dists.calculate_distances(G_bike, self.residence_coords)
        }
        """
        if self.work_coords:
            """
            self.distances_work = {
                'walk': dists.calculate_distances(G_walk, self.work_coords),
                'bike': dists.calculate_distances(G_bike, self.work_coords)
            }
            """
            # Calculate distances between residence and work
            self.distance_rw_walk = dists.calculate_distances(G_walk, self.residence_coords, self.work_coords)
            self.distance_rw_bike = dists.calculate_distances(G_bike, self.residence_coords, self.work_coords)
        else:
            self.distance_rw_walk = None
            self.distance_rw_bike = None
        
        print("Distance from residence to work:", self.distance_rw_walk)

        # Initialize distances for amenities
        self.amenities_r_walk = {}
        self.amenities_r_bike = {}
        self.distances_r_amenity_walk = {}
        self.distances_r_amenity_bike = {}
        
        if self.amenity_freqs is None:
            return

        # Get distances to the nearest amenities of that type (can be multiple) from residence location sorted by
        # how much distance they are from residence to the amenity and back to work.
        print("Initializing distances for amenities...")
        for amenity_list in self.amenity_freqs.keys():
            # Each entry is a list of distances to the nearest amenities in that list
            self.amenities_r_walk[amenity_list] = dists.get_nearest_amenities(
                G_walk, self.residence_coords, amenity_list, 10
            )
            self.distances_r_amenity_walk[amenity_list] = self.amenities_r_walk[amenity_list]["distance"].to_list()
            print(self.distances_r_amenity_walk[amenity_list])

            self.amenities_r_bike[amenity_list] = dists.get_nearest_amenities(
                G_bike, self.residence_coords, amenity_list, 10
            )
            self.distances_r_amenity_bike[amenity_list] = self.amenities_r_bike[amenity_list]["distance"].to_list()
                
        print("Initializing work distances for amenities...")
        # Initialize distances for work amenities
        if self.work_coords:
            self.amenities_w_walk = {}
            self.amenities_w_bike = {}
            self.distances_w_amenity_walk = {}
            self.distances_w_amenity_bike = {}
            
            # Get distances to the nearest amenities in each list from work location sorted by
            # how much distance they are from work to the amenity and back to residence.
            # This is done to simulate the agent's behavior of going to work and then visiting an amenity on the way home.
            for amenity_list in self.amenity_freqs.keys():
                self.amenities_w_walk[amenity_list] = dists.get_nearest_amenities_inbetween(
                    G_walk, self.work_coords, self.residence_coords, amenity_list, 10
                )
                self.distances_w_amenity_walk[amenity_list] = self.amenities_w_walk[amenity_list]["distance"].to_list()
                print(self.distances_w_amenity_walk[amenity_list])

                self.amenities_w_bike[amenity_list] = dists.get_nearest_amenities_inbetween(
                    G_bike, self.work_coords, self.residence_coords, amenity_list, 10
                )
                self.distances_w_amenity_bike[amenity_list] = self.amenities_w_bike[amenity_list]["distance"].to_list()
                
        else:
            self.distances_w_amenity_walk = None
            self.distances_w_amenity_bike = None
        

    def simulate_day(self):
        """
        Simulate a day in the life of the agent.
        """
        print("Simulating a day...")
        
        total_travel_time = 0

        travel_mode = None
        is_at_work = False

        # Check if the agent is going to bike today
        if random.random() < self.bike_freq:
            travel_mode = 'bike'
            print("Agent is biking today.")
        else:
            travel_mode = 'walk'
            print("Agent is walking today.")

        # Check if the agent is going to work today
        if random.random() < self.work_freq:
            # Travel to work
            if self.work_coords:
                self.work_travel_time = self.travel_time(self.distance_rw_walk if travel_mode == 'walk' else self.distance_rw_bike, mode=travel_mode)
                total_travel_time += self.work_travel_time
                is_at_work = True
                print(f"Travel time to work: {self.work_travel_time} minutes")
            else:
                print("No work location specified.")
        
        # Visit amenities based on frequency
        if self.amenity_freqs is None:
            print("No amenity frequencies specified.")
            return total_travel_time
        for amenity_type, freq in self.amenity_freqs.items():
            if random.random() < freq:
                # Check if the agent is at work or residence
                if is_at_work:
                    # Pick an amenity from the work location
                    for i in range(0, len(self.distances_w_amenity_walk[amenity_type] if travel_mode == 'walk' else self.distances_w_amenity_bike[amenity_type])):
                        if random.random() < self.curiosity:
                            
                            is_at_work = False  # The agent is no longer at work after visiting an amenity

                            distance_to_amenity = self.distances_w_amenity_walk[amenity_type][i] - self.distance_rw_walk if travel_mode == 'walk' else self.distances_w_amenity_bike[amenity_type][i] - self.distance_rw_bike
                            if distance_to_amenity < 1:
                                # disregard if detour is too small
                                break
                            
                            print(f"Visiting {amenity_type} from work: {distance_to_amenity}")


                            # Calculate travel time to the amenity
                            travel_time = self.travel_time(distance_to_amenity, mode=travel_mode)
                            total_travel_time += travel_time
                            print(f"Travel time to {amenity_type} from work: {travel_time} minutes")
                            self.df_travel.loc[len(self.df_travel)] = [
                                travel_time,
                                travel_mode,
                                amenity_type,
                                self.amenities_w_walk[amenity_type].iloc[i]["centroid"].coords[0] if travel_mode == 'walk' else self.amenities_w_bike[amenity_type].iloc[i]["centroid"].coords[0]
                            ]
                            
                            break

                else:
                    # Pick an amenity from the residence location
                    for i in range(0, len(self.distances_r_amenity_walk[amenity_type] if travel_mode == 'walk' else self.distances_r_amenity_bike[amenity_type])):
                        if random.random() < self.curiosity:
                            distance_to_amenity = self.distances_r_amenity_walk[amenity_type][i] if travel_mode == 'walk' else self.distances_r_amenity_bike[amenity_type][i]
                            print(f"Visiting {amenity_type} from residence: {distance_to_amenity}")

                            # Calculate travel time to the amenity
                            travel_time = self.travel_time(distance_to_amenity, mode=travel_mode)

                            total_travel_time += travel_time
                            print(f"Travel time to {amenity_type} from residence: {travel_time} minutes")
                            
                            self.df_travel.loc[len(self.df_travel)] = [
                                travel_time,
                                travel_mode,
                                amenity_type,
                                self.amenities_r_walk[amenity_type].iloc[i]["centroid"].coords[0] if travel_mode == 'walk' else self.amenities_r_bike[amenity_type].iloc[i]["centroid"].coords[0]
                            ]
                            break
        print(f"Total travel time for the day: {total_travel_time} minutes")
        return total_travel_time
    

# Example usage
rcoords = dists.residences.sample(1).geometry.values[0].centroid.coords[0]
wcoords = dists.offices.sample(1).geometry.values[0].centroid.coords[0]

agent = person(
residence_coords= rcoords, 
work_coords= wcoords,
work_freq=0.5,
walk_speed=4,
bike_speed=10,
bike_freq=0.5,
amenity_freqs={ ('cafe', 'park'): 0.5, ('school', 'restaurant'): 0.5 },
seed=42
)
for i in range(0, 10):
    print("Simulating day", i+1)
    agent.simulate_day()
print(agent.df_travel)
