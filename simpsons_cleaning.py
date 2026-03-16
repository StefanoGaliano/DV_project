import re

import pandas as pd
import numpy as np
import re
import os

os.makedirs("outputs", exist_ok=True)


df = pd.read_csv("data/simpsons_episodes.csv")


cols_to_drop = ["image_url", "video_url"]
df.drop(columns=cols_to_drop, inplace=True)


# Fix malformed titles (Season 28 rows)

# Three Season-28 titles contain leftover Wikipedia citation
# We strip the trailing quote + bracket reference.


n_fixed = 0
def clean_title(t):
    global n_fixed
    cleaned = re.sub(r'"?\[\d+\]$', '', str(t)).strip()
    if cleaned != t:
        n_fixed += 1
    return cleaned

df["title"] = df["title"].apply(clean_title)
print(f"Titles fixed (citation artifacts removed): {n_fixed}")
assert not df["title"].str.contains(r"\[\d+\]", na=False).any(), \
    "Some titles still contain citation artifacts!"


df["original_air_date"] = pd.to_datetime(df["original_air_date"], errors="coerce")
n_bad_dates = df["original_air_date"].isnull().sum()
print(f"Unparseable dates: {n_bad_dates}")


df["air_year"]         = df["original_air_date"].dt.year
df["air_month"]        = df["original_air_date"].dt.month
df["air_dayofweek_num"] = df["original_air_date"].dt.dayofweek   # 0=Mon … 6=Sun
day_map = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",
           4:"Friday",5:"Saturday",6:"Sunday"}
df["air_dayofweek"] = df["air_dayofweek_num"].map(day_map)


if "original_air_year" in df.columns:
    mismatch = (df["original_air_year"] != df["air_year"]).sum()
    print(f"Year mismatches between raw column and parsed date: {mismatch}")
    df.drop(columns=["original_air_year"], inplace=True)


print("\nMissing data in episodes:")
for col in ["imdb_rating", "imdb_votes", "us_viewers_in_millions", "views"]:
    if col in df.columns:
        print(f"  {col:<25}: {df[col].isnull().sum()}")


df["imdb_votes"] = df["imdb_votes"].astype("Int64")
df["views"]      = df["views"].astype("Int64")


assert df.duplicated(subset=["id"]).sum() == 0, "Duplicate episode IDs found!"
assert df.duplicated(subset=["season","number_in_season"]).sum() == 0, \
    "Duplicate (season, episode) pairs found!"

df.sort_values("original_air_date", inplace=True)
df.reset_index(drop=True, inplace=True)


df["imdb_rating_roll5"] = (
    df["imdb_rating"]
    .rolling(window=5, min_periods=3, center=True)
    .mean()
    .round(3)
)


season_agg = df.groupby("season").agg(
    season_avg_imdb    = ("imdb_rating",            "mean"),
    season_avg_viewers = ("us_viewers_in_millions",  "mean"),
    season_n_episodes  = ("id",                      "count"),
).round(3).reset_index()
df = df.merge(season_agg, on="season", how="left")

print(f"\nEpisodes dataset final shape: {df.shape}")
print(f"Dtypes:\n{df.dtypes.to_string()}\n")



chars = pd.read_csv("simpsons_characters.csv")


print(f"Characters raw shape: {chars.shape}")
print(f"Columns: {chars.columns.tolist()}")


chars.columns = chars.columns.str.strip().str.lower().str.replace(" ", "_")


def fix_char_name(name):
    if isinstance(name, str) and name.isupper():
        return name.title()
    return name

chars["name"] = chars["name"].apply(fix_char_name)


chars["normalized_name"] = chars["normalized_name"].str.strip().str.lower()


gender_map = {"m": "male", "f": "female"}
chars["gender"] = chars["gender"].map(gender_map)   # NaN stays NaN

n_no_gender = chars["gender"].isnull().sum()
print(f"Characters with unknown gender: {n_no_gender} "
      f"({n_no_gender/len(chars)*100:.1f}%)")


assert chars.duplicated(subset=["id"]).sum() == 0, \
    "Duplicate character IDs found!"


print(f"Characters clean shape: {chars.shape}")
print(f"Gender distribution:\n{chars['gender'].value_counts(dropna=False)}\n")



locs = pd.read_csv("simpsons_locations.csv")


print(f"Locations raw shape: {locs.shape}")
print(f"Columns: {locs.columns.tolist()}")


locs.columns = locs.columns.str.strip().str.lower().str.replace(" ", "_")


def fix_loc_name(name):
    if isinstance(name, str) and name.isupper():
        return name.title()
    return name

locs["name"] = locs["name"].apply(fix_loc_name)


locs["normalized_name"] = locs["normalized_name"].str.strip().str.lower()


n_dup_names = locs.duplicated(subset=["normalized_name"]).sum()
print(f"Duplicate location names (normalised): {n_dup_names}")
locs.drop_duplicates(subset=["normalized_name"], keep="first", inplace=True)


assert locs.duplicated(subset=["id"]).sum() == 0, \
    "Duplicate location IDs found!"

locs.reset_index(drop=True, inplace=True)
print(f"Locations clean shape: {locs.shape}\n")


episodes_out  = "outputs/simpsons_episodes_clean.csv"
chars_out     = "outputs/simpsons_characters_clean.csv"
locs_out      = "outputs/simpsons_locations_clean.csv"

df.to_csv(episodes_out, index=False)
chars.to_csv(chars_out, index=False)
locs.to_csv(locs_out, index=False)

print("Clean datasets saved:")
print(f"  {episodes_out}  — {df.shape}")
print(f"  {chars_out}     — {chars.shape}")
print(f"  {locs_out}      — {locs.shape}")
