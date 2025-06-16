# -*- coding: utf-8 -*-
import eindhovenDists3 as dists
import pandas as pd
import random


# Person class simulates the behavior of a person ("agent") in a city.
# The agent has a residence and work location, and can travel to amenities based on their preferences and travel modes.
# The agent's travel behavior is influenced by their "curiosity" and the frequency of visiting work and amenities.
# The agent can travel by walking or biking, and the travel time is calculated based on the distance to the destination and the speed of the chosen mode of transport.
# The agent's daily activities are simulated, including travel to work and visits to amenities.
# The total travel time for the day is calculated and returned.

# Slight notice: most of the actual calculations are done in the eindhovenDistsBallTree.py file
# Check that out for the actual distance calculations.

class agent:
    def __init__(self, residence, work=None, walk_speed=4.0, bike_speed=10.0, bike_freq=0.0, work_freq=1.0, features_freqs=None, curiosity_factor=1, smart=True, seed=0):
        self.residence = residence  # This should be a gdf.GeoDataFrame with a single point representing the residence
        self.work = work  # This should be a gdf.GeoDataFrame with a single point representing the work location
        self.work_freq = work_freq  # Frequency of going to work (1.0 means every day)
        self.walk_speed = walk_speed  
        self.bike_speed = bike_speed
        self.bike_freq = bike_freq # Frequency of biking (0.0 means never, 1.0 means always)
        self.features_freqs = features_freqs # Dictionary of amenity type lists and their frequencies (0.0 to 1.0)
        self.curiosity_factor = curiosity_factor # The agent will randomly pick from of the nearest curiosity_factor amenities
        self.smart = smart  # If True, the agent will optimize travel time by choosing points between work and residence to visit.
        random.seed(seed)
        self.seed = seed  # Seed for random number generation
        self.df_travel = pd.DataFrame(columns=["day", "travel_time", "mode", "amenity_type", "feature_ID"])
        self.df_days = pd.DataFrame(columns=["day", "total_travel_time", "work_travel_time"])
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
        # print("Loading graph...")
        self.graph = dists.get_data()['graph']
        # print("Graph loaded.") 

    def simulate_day(self, start_location=None, end_location=None, starting_travel_time = 0, sampled_amenities = None): # Locations should be gdf entries
        """
        Simulate a day in the life of the agent.
        """
        
        if start_location is None:
            # If no location is provided, start at the residence's nearest node
            location = self.residence['nearest_node'].iloc[0]  # Start at the residence's nearest node
        else:
            location = start_location['nearest_node'].iloc[0]  # Use the provided start location's nearest node

        
        if end_location is None:
            # If no end location is provided, end at the residence's nearest node
            # print("No end location provided, ending at residence.")
            end_location = self.residence

        end_node = end_location['nearest_node'].iloc[0]  # Use the provided end location's nearest node

        total_travel_time = starting_travel_time  # Start with the provided travel time, if any
        work_travel_time = 0  # Initialize work travel time

        travel_mode = None

        # Check if the agent is going to bike today
        if random.random() < self.bike_freq:
            travel_mode = 'bike'
            # print("Agent is biking today.")
        else:
            travel_mode = 'walk'
            # print("Agent is walking today.")

        work_node = self.work['nearest_node'].iloc[0] if self.work is not None else None

        # Check if the agent is going to work today
        if random.random() < self.work_freq:
            # Travel to work
            if self.work is not None:
                # print("Agent is going to work today.")
                work_travel_time = self.travel_time(
                    dists.get_distances(self.graph, location, self.work['nearest_node'].iloc[0]), mode=travel_mode)
                total_travel_time += work_travel_time
                location = work_node # Update location to work's nearest node
                self.df_travel.loc[len(self.df_travel)] = [
                    len(self.df_days) + 1,  # day number
                    work_travel_time,
                    travel_mode,
                    'work',
                    self.work.index[0]  # Use the index of the work location as feature_ID
                ]
                # print(f"Travel time to work: {work_travel_time} minutes")
            else:
                print("No work location specified.")
        
        if sampled_amenities is None:
            # Visit amenities based on frequency
            if self.features_freqs is None:
                print("No amenity frequencies specified.")
                return
            
            # Sample amenities based on their frequencies
            sampled_amenities = random.choices(
                list(self.features_freqs.keys()), 
                weights=list(self.features_freqs.values()), 
                k=1
            )

        for amenity_type in sampled_amenities:
            # Check if the agent is smart and this is the last amenity to visit
            if amenity_type == sampled_amenities[-1] and self.smart: # Check if it is the last amenity to visit
                
                amenities = dists.get_nearest_amenities_from_node(
                    self.graph, location, amenity_type, max_count=self.curiosity_factor
                )
                i = random.randint(0, len(amenities) - 1)
                
                distance_to_amenity = amenities.iloc[i]["distance"] - dists.get_distances(
                    self.graph, end_node, location
                )

                self.location = end_node

                if distance_to_amenity < 1:
                    # disregard if detour is too small
                    break
                
                # print(f"Visiting {amenity_type} on the way home: {distance_to_amenity}")


                # Calculate travel time to the amenity
                travel_time = self.travel_time(distance_to_amenity, mode=travel_mode)
                total_travel_time += travel_time
                # print(f"Travel time to {amenity_type} from work: {travel_time} minutes")
                self.df_travel.loc[len(self.df_travel)] = [
                    len(self.df_days) + 1,  # day number
                    travel_time,
                    travel_mode,
                    amenity_type,
                    amenities.index[i]  # Use the index of the amenity as feature_ID
                ]
                
                break

            else:
                # print(f"Visiting {amenity_type} from current location: {location}")
                # Pick an amenity from the current location
                
                amenities = dists.get_nearest_amenities_from_node(
                    self.graph, location, amenity_type, max_count=self.curiosity_factor
                )
                i = random.randint(0, len(amenities) - 1)
                
                distance_to_amenity = amenities.iloc[i]["distance"]
                # print(f"Visiting {amenity_type}: {distance_to_amenity} meters")

                # Calculate travel time to the amenity
                travel_time = self.travel_time(distance_to_amenity, mode=travel_mode)

                total_travel_time += travel_time
                # print(f"Travel time to {amenity_type}: {travel_time} minutes")

                
                self.location = amenities.iloc[i]["centroid"].coords[0]
                
                self.df_travel.loc[len(self.df_travel)] = [
                    len(self.df_days) + 1,  # day number
                    travel_time,
                    travel_mode,
                    amenity_type,
                    amenities.index[i]
                ]
                break
        # Return to residence if not already there
        if location != end_node:
            # print("Returning to residence...")
            distance_to_residence = dists.get_distances(self.graph, location, end_node)
            travel_time = self.travel_time(distance_to_residence, mode=travel_mode)
            total_travel_time += travel_time
            # print(f"Travel time back to residence: {travel_time} minutes")
            self.df_travel.loc[len(self.df_travel)] = [
                len(self.df_days) + 1,  # day number
                travel_time,
                travel_mode,
                'residence' if end_location.iloc[0]['main_tag'] != 'hub' else 'hub',
                end_location.index[0]  # Use the index of the residence as feature_ID
            ]
        # print(f"Total travel time for the day: {total_travel_time} minutes")
        # Flatten the sampled amenities list into a string
        self.df_days.loc[len(self.df_days)] = [
            len(self.df_days) + 1,
            total_travel_time,
            work_travel_time,
        ]

        # Return the last row of df_days and df_travel for the day
        # print("Day simulation complete.")
        return self.df_travel[self.df_travel['day'] == len(self.df_days)], self.df_days.iloc[-1]
    
    def simulate_day_with_hubs(self, sampled_amenities=None):
        """
        Simulate a day in the life of the agent with hubs.
        This method is similar to simulate_day but includes logic for visiting hubs.
        """
        # print("Simulating a day with hubs...")
        
        travel_mode = 'walk' # Assuming the agent will travel to a hub, they will walk

        # Check if there are hubs available in features
        # Get the nearest hub to the residence
        nearest_hub = dists.get_nearest_amenities_from_node(
            self.graph, self.residence['nearest_node'].iloc[0], ('hub',), 1
        )

        if nearest_hub.empty:
            print("No hubs available near the residence.")
            self.simulate_day(start_location=self.residence, end_location=self.residence)
            return
        
        # Otherwise, get the distance between the residence and the hub
        hub_distance = nearest_hub['distance'].iloc[0]
        hub_travel_time = self.travel_time(hub_distance, mode=travel_mode)

        # print(f"Nearest hub found: {nearest_hub.iloc[0]['centroid'].coords[0]} with distance {hub_distance} meters and travel time {hub_travel_time} minutes.")
        # Track the travel to the hub
        self.df_travel.loc[len(self.df_travel)] = [
            len(self.df_days + 1),  # day number
            hub_travel_time,
            travel_mode,
            'hub',
            nearest_hub.index[0]  # Use the index of the hub as feature_ID
        ]

        # Initialize total travel time for the day
        total_travel_time = hub_travel_time  # Start with the travel time to the hub

        # Save the innate biking frequency
        original_bike_freq = self.bike_freq
        # Set the biking frequency to 1 after the hub visit
        self.bike_freq = 1.0

        # Simulate day from and back to hub
        self.simulate_day(start_location=nearest_hub, end_location=nearest_hub, starting_travel_time=hub_travel_time, sampled_amenities=sampled_amenities)

        # Restore the original biking frequency
        self.bike_freq = original_bike_freq

        # Add hub travel time to total travel time
        total_travel_time += hub_travel_time * 2  # Round trip to the hub
        # Add hub travel time to the travel dataframe
        self.df_travel.loc[len(self.df_travel)] = [
            len(self.df_days),  # day number
            hub_travel_time,
            travel_mode,
            'residence',
            nearest_hub.index[0]  # Use the index of the hub as feature_ID
        ]
        # Update the total travel time for the day
        self.df_days.loc[len(self.df_days) - 1] = [
            len(self.df_days),  # day number
            total_travel_time + self.df_days['total_travel_time'].iloc[-1],  # add final hub travel time to total travel time
            self.df_days.loc[len(self.df_days) - 1, 'work_travel_time'],  # work travel time remains the same
        ]

        # Return the last row of df_days and df_travel for the day
        print("Day with hubs simulation complete.")
        return self.df_travel[self.df_travel['day'] == len(self.df_days)], self.df_days.iloc[-1]

    def simulate_comparison(self, days=1, match_sampled_amenities=True):
        """
        Simulate multiple days of the agent's travel behavior.
        """
        df_travel_hubs = pd.DataFrame(columns=self.df_travel.columns)
        df_days_hubs = pd.DataFrame(columns=self.df_days.columns)
        df_travel_nohubs = pd.DataFrame(columns=self.df_travel.columns)
        df_days_nohubs = pd.DataFrame(columns=self.df_days.columns)

        for day in range(1, days + 1):
            # print(f"Simulating day {day}...")
            if match_sampled_amenities:
                # Visit amenities based on frequency
                if self.features_freqs is None:
                    print("No amenity frequencies specified.")
                    return

                # Sample amenities based on their frequencies
                sampled_amenities = random.choices(
                    list(self.features_freqs.keys()), 
                    weights=list(self.features_freqs.values()), 
                    k=1
                )
                rows_hubs = self.simulate_day_with_hubs(sampled_amenities=sampled_amenities)
                rows_nohubs = self.simulate_day(starting_travel_time=0, sampled_amenities=sampled_amenities)
            else:
                rows_hubs = self.simulate_day_with_hubs()
                rows_nohubs = self.simulate_day(starting_travel_time=0)

            # Check if the simulation returned valid results
            if rows_nohubs is None and rows_hubs is None:
                print("Both simulate_day and simulate_day_with_hubs returned None.")
                continue

            if rows_nohubs is None:
                print("simulate_day returned None.")
                continue

            if rows_hubs is None:
                print("simulate_day_with_hubs returned None.")
                continue

            rows_hubs_travel, rows_hubs_days = rows_hubs
            rows_nohubs_travel, rows_nohubs_days = rows_nohubs

            # Append the results to the respective DataFrames
            df_travel_hubs = pd.concat([df_travel_hubs, rows_hubs_travel], ignore_index=True)
            df_days_hubs.loc[len(df_days_hubs)] = rows_hubs_days
            df_travel_nohubs = pd.concat([df_travel_nohubs, rows_nohubs_travel], ignore_index=True)
            df_days_nohubs.loc[len(df_days_nohubs)] = rows_nohubs_days



        # print("Simulation complete.")
        # Reset the index of the DataFrames
        df_travel_hubs.reset_index(drop=True, inplace=True)
        df_days_hubs.reset_index(drop=True, inplace=True)
        df_travel_nohubs.reset_index(drop=True, inplace=True)
        df_days_nohubs.reset_index(drop=True, inplace=True)
        # Return the DataFrames for travel and days with and without hubs
        return {
            'travel_hubs': df_travel_hubs,
            'days_hubs': df_days_hubs,
            'travel_nohubs': df_travel_nohubs,
            'days_nohubs': df_days_nohubs
        }