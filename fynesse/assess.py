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

    # --- Monthly counts ---
    monthly_counts = gdf.groupby(['year', 'month']).size().reset_index(name='count')
    years = monthly_counts['year'].unique()

    for year in years:
        monthly_counts_year = monthly_counts[monthly_counts['year'] == year]
        # reindex months to ensure all 12 months appear
        monthly_counts_year = monthly_counts_year.set_index('month').reindex(range(1, 13), fill_value=0)

        monthly_counts_year.plot(kind='bar', y='count', legend=False)
        plt.title(f"Monthly crash counts for {year}")
        plt.xlabel("Month")
        plt.ylabel("Number of crashes")
        plt.xticks(range(0, 12), range(1, 13), rotation=0)
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
    plt.imshow(pivot, aspect='auto', cmap="Reds")
    plt.title("Crash Heatmap: Day of Week vs Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week (0=Mon, 6=Sun)")
    plt.colorbar(label="Number of crashes")
    plt.show()