"""
Address module for the fynesse framework.

This module handles question addressing functionality including:
- Statistical analysis
- Predictive modeling
- Data visualization for decision-making
- Dashboard creation
"""

from typing import Any, Union
import pandas as pd
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Here are some of the imports we might expect
# import sklearn.model_selection  as ms
# import sklearn.linear_model as lm
# import sklearn.svm as svm
# import sklearn.naive_bayes as naive_bayes
# import sklearn.tree as tree

# import GPy
# import torch
# import tensorflow as tf

# Or if it's a statistical analysis
# import scipy.stats


def analyze_data(data: Union[pd.DataFrame, Any]) -> dict[str, Any]:
    """
    Address a particular question that arises from the data.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR ANALYSIS CODE:
       - Perform statistical analysis on the data
       - Create visualizations to explore patterns
       - Build models to answer specific questions
       - Generate insights and recommendations

    2. ADD ERROR HANDLING:
       - Check if input data is valid and sufficient
       - Handle analysis failures gracefully
       - Validate analysis results

    3. ADD BASIC LOGGING:
       - Log analysis steps and progress
       - Log key findings and insights
       - Log any issues encountered

    4. EXAMPLE IMPLEMENTATION:
       if data is None or len(data) == 0:
           print("Error: No data available for analysis")
           return {}

       print("Starting data analysis...")
       # Your analysis code here
       results = {"sample_size": len(data), "analysis_complete": True}
       return results
    """
    logger.info("Starting data analysis")

    # Validate input data
    if data is None:
        logger.error("No data provided for analysis")
        print("Error: No data available for analysis")
        return {"error": "No data provided"}

    if len(data) == 0:
        logger.error("Empty dataset provided for analysis")
        print("Error: Empty dataset provided for analysis")
        return {"error": "Empty dataset"}

    logger.info(f"Analyzing data with {len(data)} rows, {len(data.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your analysis code here

        # Example: Basic data summary
        results = {
            "sample_size": len(data),
            "columns": list(data.columns),
            "data_types": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "analysis_complete": True,
        }

        # Example: Basic statistics (students should customize this)
        numeric_columns = data.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            results["numeric_summary"] = data[numeric_columns].describe().to_dict()

        logger.info("Data analysis completed successfully")
        print(f"Analysis completed. Sample size: {len(data)}")

        return results

    except Exception as e:
        logger.error(f"Error during data analysis: {e}")
        print(f"Error analyzing data: {e}")
        return {"error": str(e)}

import geopandas as gpd
import folium
from folium.plugins import MarkerCluster

def plot_crash_map(gdf, country_name="Kenya"):
    """
    Plots crash locations on a Folium map with clustering,
    overlaying the boundary of a given country.

    Parameters:
        gdf (GeoDataFrame or DataFrame): Must have 'latitude', 'longitude', 
                                         'crash_id', and 'crash_datetime'.
        country_name (str): Country to highlight (default = 'Kenya').

    Returns:
        folium.Map: Interactive map with crashes plotted.
    """
    # --- Download world boundaries 
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)

    country = world[world["NAME"] == country_name]

    if country.empty:
        raise ValueError(f"Country '{country_name}' not found in shapefile.")

    
    minx, miny, maxx, maxy = country.total_bounds

    # Create Folium map centered on crashes
    m = folium.Map(location=[gdf['latitude'].mean(), gdf['longitude'].mean()], zoom_start=6)
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    
    folium.GeoJson(country.geometry, name=f"{country_name} Boundary").add_to(m)

    # Add clustered crash markers
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in gdf.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Crash ID: {row['crash_id']}<br>Datetime: {row['crash_datetime']}"
        ).add_to(marker_cluster)

    return m


import osmnx as ox
import pandas as pd

def assign_crash_risk_to_edges(G, gdf, lon_col="longitude", lat_col="latitude"):
    """
    Assigns crash risk  to edges in a road network graph.
    
    Parameters
    ----------
    G : networkx.MultiDiGraph
        Road network graph from osmnx.
    gdf : GeoDataFrame
        Crash data with latitude/longitude columns.
    lon_col : str
        Name of the longitude column in gdf.
    lat_col : str
        Name of the latitude column in gdf.
    
    Returns
    -------
    gdf : GeoDataFrame
        Crash dataframe with added u, v, key columns (nearest edge IDs).
    G : networkx.MultiDiGraph
        Road graph with crash_count and risk attributes added to edges.
    """
    
    # --- Step 1: Find nearest edges for crash points ---
    try:
        nearest_edges = ox.distance.nearest_edges(
            G,
            X=gdf[lon_col].values,
            Y=gdf[lat_col].values,
            return_dist=False
        )
    except Exception:
        
        nearest_edges = ox.nearest_edges(G, gdf[lon_col].values, Y=gdf[lat_col].values)

    # Convert tuples -> list so pandas can split into columns
    nearest_edges_list = [list(edge) for edge in nearest_edges]
    gdf[['u', 'v', 'key']] = nearest_edges_list

    # --- Step 2: Count crashes per edge 
    edge_crash_counts = gdf.groupby(['u', 'v', 'key']).size().reset_index(name='crash_count')

    # Make a dictionary for fast lookup
    edge_risk_dict = edge_crash_counts.set_index(['u', 'v', 'key'])['crash_count'].to_dict()

    # --- Step 3: Assign crash_count & risk to graph edges 
    for u, v, k, data in G.edges(keys=True, data=True):
        risk_val = edge_risk_dict.get((u, v, k), 0)
        data['crash_count'] = risk_val
        data['risk'] = risk_val   # same as crash_count for now

    print("✅ Assigned risk values to graph edges.")
    return gdf, G


import osmnx as ox
import networkx as nx
import folium

def plot_routes_in_nairobi(length_weight=0.1):
    """
    Asks the user for start and destination in Nairobi,
    computes shortest vs safer routes, and plots them on a Folium map.
    """
    print("Downloading Nairobi road network...")
    G = ox.graph_from_place("Nairobi, Kenya", network_type="drive")
    print("Graph loaded!")

    # --- Safer route function ---
    def safer_route(G, origin_node, destination_node, weight='risk', length_weight=0.1):
        def custom_weight(u, v, data):
            risk = data.get(weight, 0)
            length = data.get("length", 0)
            return risk + length_weight * length
        try:
            return nx.shortest_path(G, source=origin_node, target=destination_node, weight=custom_weight)
        except nx.NetworkXNoPath:
            print("No safe path found.")
            return None

    # --- Helper: add route to map ---
    def add_route_to_map(G, route_nodes, fmap, color="blue", weight=5, popup="Route"):
        route_edges = list(zip(route_nodes[:-1], route_nodes[1:]))
        coords = []
        for u, v in route_edges:
            data = min(G.get_edge_data(u, v).values(), key=lambda d: d.get("length", 0))
            if "geometry" in data:
                xs, ys = data["geometry"].xy
                coords.extend([(y, x) for x, y in zip(xs, ys)])
            else:
                coords.extend([(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])])
        folium.PolyLine(coords, color=color, weight=weight, opacity=0.8, popup=popup).add_to(fmap)

    # --- User input ---
    start_place = input("Enter START location: ")
    end_place = input("Enter DESTINATION location: ")

    # --- Geocode ---
    start_point = ox.geocode(start_place + ", Nairobi, Kenya")
    end_point = ox.geocode(end_place + ", Nairobi, Kenya")
    print(f"Start coordinates: {start_point}")
    print(f"Destination coordinates: {end_point}")

    # --- Nearest graph nodes ---
    origin_node = ox.distance.nearest_nodes(G, X=start_point[1], Y=start_point[0])
    destination_node = ox.distance.nearest_nodes(G, X=end_point[1], Y=end_point[0])

    # --- Compute routes ---
    try:
        shortest_path_nodes = nx.shortest_path(G, source=origin_node, target=destination_node, weight="length")
    except nx.NetworkXNoPath:
        shortest_path_nodes = None
        print("No shortest path found!")

    safer_path_nodes = safer_route(G, origin_node, destination_node, length_weight=length_weight)

    # --- Folium Map ---
    m = folium.Map(location=start_point, zoom_start=13)

    if shortest_path_nodes:
        add_route_to_map(G, shortest_path_nodes, m, color="blue", popup="Shortest Route")

    if safer_path_nodes:
        add_route_to_map(G, safer_path_nodes, m, color="red", popup="SAFE Route 🚦")

    # --- Add markers ---
    folium.Marker(start_point, popup="Start: " + start_place, icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(end_point, popup="Destination: " + end_place, icon=folium.Icon(color="red")).add_to(m)

    print("\n✅ Routes plotted successfully! Blue = shortest, Red = safe")
    return m
