"""
VibeMatch — Data Exploration Script (v2)
Run: python explore_data.py
"""
import pandas as pd

files = ['games', 'tags', 'genres', 'categories', 'promotional']
separator = "=" * 60

for f in files:
    path = f"../data/raw/steam-insights/{f}.csv"
    print(f"\n{separator}")
    try:
        # on_bad_lines='skip' ignores rows with inconsistent column counts
        df = pd.read_csv(path, on_bad_lines='skip')
        print(f"{f.upper()}.csv -- {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        print(separator)
        print(df.head(3).to_string())
    except Exception as e:
        print(f"{f.upper()}.csv -- ERROR: {e}")
    print()

# Deeper analysis
print(f"\n{separator}")
print("DETAILED ANALYSIS")
print(separator)

# Games table
games = pd.read_csv("../data/raw/steam-insights/games.csv", on_bad_lines='skip')
print(f"\nGAMES -- unique app IDs: {games.iloc[:, 0].nunique()}")
print(f"\nNull counts per column:")
print(games.isnull().sum().to_string())

# Tags table
tags = pd.read_csv("../data/raw/steam-insights/tags.csv", on_bad_lines='skip')
print(f"\nTAGS -- unique app IDs: {tags.iloc[:, 0].nunique()}")
if len(tags.columns) > 1:
    tag_col = tags.columns[1]
    print(f"Unique tag names: {tags[tag_col].nunique()}")
    print(f"\nTop 20 most common tags:")
    print(tags[tag_col].value_counts().head(20).to_string())

# Promotional table
promo = pd.read_csv("../data/raw/steam-insights/promotional.csv", on_bad_lines='skip')
print(f"\nPROMOTIONAL -- unique app IDs: {promo.iloc[:, 0].nunique()}")
print(f"Columns: {list(promo.columns)}")
print(f"\nFirst 5 rows:")
print(promo.head(5).to_string())

# Check how many games have screenshots
screenshot_cols = [c for c in promo.columns if 'screenshot' in c.lower() or 'image' in c.lower() or 'url' in c.lower() or 'photo' in c.lower()]
print(f"\nPossible screenshot columns: {screenshot_cols}")
if not screenshot_cols:
    print("No obvious screenshot columns found. Showing all column samples:")
    for col in promo.columns:
        sample = promo[col].dropna().head(2).tolist()
        print(f"  {col}: {sample}")

# Cross-reference: how many games have BOTH tags and promotional data?
game_ids_with_tags = set(tags.iloc[:, 0].unique())
game_ids_with_promo = set(promo.iloc[:, 0].unique())
game_ids_games = set(games.iloc[:, 0].unique())
overlap = game_ids_with_tags & game_ids_with_promo & game_ids_games
print(f"\nCROSS-REFERENCE:")
print(f"  Games in games.csv: {len(game_ids_games)}")
print(f"  Games with tags: {len(game_ids_with_tags)}")
print(f"  Games with promotional data: {len(game_ids_with_promo)}")
print(f"  Games in ALL three tables: {len(overlap)}")