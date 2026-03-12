import pandas as pd
import numpy as np


df = pd.read_csv("simpsons_episodes.csv")


# Drop  non-analytical columns
# image_url and video_url are not needed for analysis.

cols_to_drop = ["image_url", "video_url"]
df.drop(columns=cols_to_drop, inplace=True)


# Fix malformed titles (Season 28 rows)

# Three Season-28 titles contain leftover Wikipedia citation
# We strip the trailing quote + bracket reference.

import re

n_fixed = 0
def clean_title(t):
    global n_fixed
    cleaned = re.sub(r'"?\[\d+\]$', '', str(t)).strip()
    if cleaned != t:
        n_fixed += 1
    return cleaned

df["title"] = df["title"].apply(clean_title)

# Verify
assert not df["title"].str.contains(r"\[\d+\]", na=False).any(), \
    "Some titles still contain citation artifacts!"


# Parse and validate original_air_date

df["original_air_date"] = pd.to_datetime(df["original_air_date"], errors="coerce")
n_bad_dates = df["original_air_date"].isnull().sum()

# Derive useful temporal columns
df["air_year"]    = df["original_air_date"].dt.year
df["air_month"]   = df["original_air_date"].dt.month
df["air_dayofweek_num"]  = df["original_air_date"].dt.dayofweek   
day_map = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",
           4:"Friday",5:"Saturday",6:"Sunday"}
df["air_dayofweek"] = df["air_dayofweek_num"].map(day_map)

# Sanity check to see if original_air_year matches parsed year
mismatch = (df["original_air_year"] != df["air_year"]).sum()

# We drop original_air_year as it is redundant
df.drop(columns=["original_air_year"], inplace=True)


# Handle missing numeric values
# We do NOT impute, yet we flag them as NaN 
print(f"Missing data")
print(f"imdb_rating            : {df['imdb_rating'].isnull().sum()}")
print(f"imdb_votes             : {df['imdb_votes'].isnull().sum()}")
print(f"us_viewers_in_millions : {df['us_viewers_in_millions'].isnull().sum()}")
print(f"views                  : {df['views'].isnull().sum()}")


#Uncomment when we're sure we need to drop season 28
#df = df[df['season'] != 28]


# Cast / tidy dtypes

# imdb_votes: stored as float (due to NaN); cast to Int64 (nullable int)
df["imdb_votes"] = df["imdb_votes"].astype("Int64")
df["views"]      = df["views"].astype("Int64")

# season and number_in_season are already int64
print(f"Dtypes after cast:\n{df.dtypes.to_string()}")


# Verify uniqueness & ordering

assert df.duplicated(subset=["id"]).sum() == 0, "Duplicate IDs found!"
assert df.duplicated(subset=["season","number_in_season"]).sum() == 0, \
    "Duplicate (season, episode) pairs found!"

# Sort by air date for natural ordering
df.sort_values("original_air_date", inplace=True)
df.reset_index(drop=True, inplace=True)


# Add convenience columns for analysis

# Rolling 5-episode IMDB rating (for smooth trend chart)
df["imdb_rating_roll5"] = (
    df["imdb_rating"]
    .rolling(window=5, min_periods=3, center=True)
    .mean()
    .round(3)
)

# Season-level aggregates (merged back for per-episode rows)
season_agg = df.groupby("season").agg(
    season_avg_imdb   = ("imdb_rating",            "mean"),
    season_avg_viewers= ("us_viewers_in_millions",  "mean"),
    season_n_episodes = ("id",                      "count"),
).round(3).reset_index()
df = df.merge(season_agg, on="season", how="left")


#  — Final shape & save

output_path = "outputs/simpsons_episodes_clean.csv"
#df.to_csv(output_path, index=False)
print(f"Clean dataset saved to: {output_path}")
