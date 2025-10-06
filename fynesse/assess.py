from typing import Any, Union
import pandas as pd
import logging

from .config import *
from . import access
import matplotlib.pyplot as plt
import math
import osmnx as ox

# Set up logging
logger = logging.getLogger(__name__)

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn

import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded.
How are missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete visualisation
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct
and correctly timezoned."""


def data() -> Union[pd.DataFrame, Any]:
    """
    Load the data from access and ensure missing values are correctly encoded as well as
    indices correct, column names informative, date and times correctly formatted.
    Return a structured data structure such as a data frame.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR DATA ASSESSMENT CODE:
       - Load data using the access module
       - Check for missing values and handle them appropriately
       - Validate data types and formats
       - Clean and prepare data for analysis

    2. ADD ERROR HANDLING:
       - Handle cases where access.data() returns None
       - Check for data quality issues
       - Validate data structure and content

    3. ADD BASIC LOGGING:
       - Log data quality issues found
       - Log cleaning operations performed
       - Log final data summary

    4. EXAMPLE IMPLEMENTATION:
       df = access.data()
       if df is None:
           print("Error: No data available from access module")
           return None

       print(f"Assessing data quality for {len(df)} rows...")
       # Your data assessment code here
       return df
    """
    logger.info("Starting data assessment")

    # Load data from access module
    df = access.data()

    # Check if data was loaded successfully
    if df is None:
        logger.error("No data available from access module")
        print("Error: Could not load data from access module")
        return None

    logger.info(f"Assessing data quality for {len(df)} rows, {len(df.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your data assessment code here

        # Example: Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts.to_dict()}")
            print(f"Missing values found: {missing_counts.sum()} total")

        # Example: Check data types
        logger.info(f"Data types: {df.dtypes.to_dict()}")

        # Example: Basic data cleaning (students should customize this)
        # Remove completely empty rows
        df_cleaned = df.dropna(how="all")
        if len(df_cleaned) < len(df):
            logger.info(f"Removed {len(df) - len(df_cleaned)} completely empty rows")

        logger.info(f"Data assessment completed. Final shape: {df_cleaned.shape}")
        return df_cleaned

    except Exception as e:
        logger.error(f"Error during data assessment: {e}")
        print(f"Error assessing data: {e}")
        return None

def plot_city_map(place_name=None, latitude=None, longitude=None,
                  box_width= 0.1, box_height= 0.1, tags=None):
   
    
   north = latitude + box_height/2
   south = latitude - box_height/2
   west = longitude - box_width/2
   east = longitude + box_width/2
   bbox = (west, south, east, north)


   tags = tags or {"amenity": True}
   pois = ox.features_from_bbox(bbox, tags)

   graph = ox.graph_from_bbox(bbox)
   # City area
   area = ox.geocode_to_gdf(place_name)
   # Street network
   nodes, edges = ox.graph_to_gdfs(graph)
   # Buildings
   buildings = ox.features_from_bbox(bbox, tags={"building": True})

   fig, ax = plt.subplots(figsize=(6,6))
   area.plot(ax=ax, color="tan", alpha=0.5)
   buildings.plot(ax=ax, facecolor="gray", edgecolor="gray")
   edges.plot(ax=ax, linewidth=1, edgecolor="black", alpha=0.3)
   nodes.plot(ax=ax, color="black", markersize=1, alpha=0.3)
   pois.plot(ax=ax, color="green", markersize=5, alpha=1)
   ax.set_xlim(west, east)
   ax.set_ylim(south, north)
   ax.set_title(place_name, fontsize=14)
   plt.show()
   return {"place_name": place_name, "lat": latitude, "lon": longitude}


def query(data: Union[pd.DataFrame, Any]) -> str:
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data: Union[pd.DataFrame, Any]) -> None:
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError



import matplotlib.pyplot as plt

def plot_crash_trends(gdf):
    """
    Plots total crash counts per year and monthly crash counts for each year.

    Parameters:
        gdf (GeoDataFrame or DataFrame): Must contain 'year' and 'month' columns.
    """

    # --- Yearly counts ---
    yearly_counts = gdf.groupby('year').size().reset_index(name='count')
    yearly_counts.plot(x='year', y='count', marker='o', linestyle='-')
    plt.title("Total crash counts per year")
    plt.xlabel("Year")
    plt.ylabel("Number of crashes")
    plt.show()

   

def plot_monthly_totals(gdf):
    """
    Plots total crash counts by month aggregated across all years.

    Parameters:
        gdf (GeoDataFrame or DataFrame): Must contain 'month' column (1–12).
    """

    month_totals = gdf['month'].value_counts().sort_index()

    month_totals.plot(kind='bar', legend=False)
    plt.title("Total crash counts by month (all years)")
    plt.xlabel("Month")
    plt.ylabel("Number of crashes")
    plt.xticks(range(0, 12), range(1, 13), rotation=0)
    plt.show()


def plot_hourly_patterns(gdf):
    """
    Plots:
    1. Bar chart of crashes by hour of the day
    2. Heatmap of crashes (day of week vs hour)

    Parameters:
        gdf (GeoDataFrame or DataFrame): Must contain 'hour' and 'dayofweek' columns.
            - 'hour' should be 0–23
            - 'dayofweek' should be 0=Mon, ..., 6=Sun
    """

    # --- Crashes by hour of the day ---
    hour_counts = gdf['hour'].value_counts().sort_index()
    hour_counts.plot(kind='bar')
    plt.title("Crashes by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Crash Count")
    plt.show()

    # --- Heatmap: dayofweek vs hour ---
    pivot = gdf.groupby(['dayofweek', 'hour']).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 6))
    plt.imshow(pivot, aspect='auto')
    plt.title("Crash Heatmap: Day of Week vs Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week (0=Mon, 6=Sun)")
    plt.colorbar(label="Number of crashes")
    plt.show()

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
import geopandas as gpd
import folium
import ipywidgets as widgets
from IPython.display import display, IFrame
import os

def risk_visualizer(gdf, lon_col="longitude", lat_col="latitude", cache_dir="cached_graphs"):
    """
    Interactive crash risk visualization by county using Folium with a color legend.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Crash dataset with latitude/longitude.
    lon_col : str
        Column name for longitude in gdf.
    lat_col : str
        Column name for latitude in gdf.
    cache_dir : str
        Directory to cache graphml files to avoid redownloading.
    """
    os.makedirs(cache_dir, exist_ok=True)

    counties = ["Nairobi, Kenya", "Machakos, Kenya", "Kiambu, Kenya", "Murang'a, Kenya", "Kajiado, Kenya"]

    dropdown = widgets.Dropdown(
        options=[("Select a county...", None)] + [(c, c) for c in counties],
        description="Select County:",
        value="Nairobi, Kenya",   
        style={'description_width': 'initial'},
        layout=widgets.Layout(width="50%")
    )

    def process_county(county_name):
        if county_name is None:
            print("defaulting to Nairobi, Kenya.\n")
            county_name = "Nairobi, Kenya"

        # --- caching logic ---
        safe_name = county_name.replace(" ", "_").replace(",", "")
        graph_path = os.path.join(cache_dir, f"{safe_name}.graphml")

        if os.path.exists(graph_path):
            print(f"Loading cached graph for {county_name}...")
            G = ox.load_graphml(graph_path)
        else:
            print(f" Downloading road network for {county_name}...")
            G = ox.graph_from_place(county_name, network_type="drive")
            ox.save_graphml(G, graph_path)
            print(f"Graph cached at {graph_path}")

        # --- crash mapping ---
        try:
            nearest_edges = ox.distance.nearest_edges(
                G,
                X=gdf[lon_col].values,
                Y=gdf[lat_col].values,
                return_dist=False
            )
        except Exception:
            nearest_edges = ox.nearest_edges(
                G,
                gdf[lon_col].values,
                Y=gdf[lat_col].values
            )

        nearest_edges_list = [list(edge) for edge in nearest_edges]
        gdf[['u', 'v', 'key']] = nearest_edges_list

        edge_crash_counts = gdf.groupby(['u', 'v', 'key']).size().reset_index(name='crash_count')
        edge_risk_dict = edge_crash_counts.set_index(['u', 'v', 'key'])['crash_count'].to_dict()

        for u, v, k, data in G.edges(keys=True, data=True):
            risk_val = edge_risk_dict.get((u, v, k), 0)
            data['crash_count'] = risk_val
            data['risk'] = risk_val

        edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()

        print("\n Roads with crash counts & risk factors:")
        if "name" in edges_gdf.columns:
            edges_gdf["name"] = edges_gdf["name"].fillna("Unnamed road")
            display(edges_gdf[['u','v','key','name','crash_count','risk']].sort_values("risk", ascending=False).head(20))
        else:
            display(edges_gdf[['u','v','key','crash_count','risk']].sort_values("risk", ascending=False).head(20))

        

    widgets.interact(process_county, county_name=dropdown)

