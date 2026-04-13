"""
VibeMatch — Step 2: Data Preprocessing (v6 — final)
Run from project root: python src/data/preprocess.py
Output: data/processed/games_master.csv
"""
import pandas as pd
import re
from pathlib import Path


def parse_promotional(filepath):
    """
    Parse promotional.csv — each line is one game.
    Extract app_id and screenshot URLs using regex.
    
    The URLs in the file have escaped backslashes before quotes,
    so we use a pattern that stops at backslash, quote, or whitespace.
    We filter out header_image and page_bg URLs to keep only screenshots.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    print(f"  Total lines in file: {len(lines) - 1}")
    
    results = []
    
    for line in lines[1:]:  # Skip header
        # Extract app_id
        app_match = re.match(r'^"(\d+)"', line)
        if not app_match:
            continue
        app_id = int(app_match.group(1))
        
        # Extract all steamstatic .jpg URLs (loose pattern that handles escaping)
        all_urls = re.findall(
            r'https://shared\.akamai\.steamstatic\.com/[^\s"\\]+\.jpg',
            line
        )
        
        # Filter: keep only screenshot URLs (exclude header images and backgrounds)
        screenshot_urls = [
            u for u in all_urls
            if '/header.' not in u and '/page_bg' not in u and '/capsule' not in u
        ]
        
        # Prefer 1920x1080, fall back to 600x338
        full_res = [u for u in screenshot_urls if '1920x1080' in u]
        thumbnails = [u for u in screenshot_urls if '600x338' in u]
        
        # Use full-res if available, otherwise thumbnails
        urls = full_res if len(full_res) >= 3 else thumbnails
        
        if len(urls) >= 3:
            results.append({
                'app_id': app_id,
                'url_0': urls[0],
                'url_1': urls[1],
                'url_2': urls[2],
                'screenshot_count': len(urls),
            })
    
    return pd.DataFrame(results)


def parse_games_names(filepath):
    """
    Parse games.csv to extract app_id and game name.
    Uses raw text parsing to handle commas in JSON fields.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Each record line starts with "app_id","name"
    matches = re.findall(r'^"(\d+)","([^"]*)"', content, re.MULTILINE)
    
    results = []
    for app_id_str, name in matches:
        try:
            results.append({'app_id': int(app_id_str), 'name': name})
        except ValueError:
            continue
    
    return pd.DataFrame(results).drop_duplicates(subset='app_id')


def main():
    print("=" * 60)
    print("VibeMatch — Data Preprocessing")
    print("=" * 60)
    
    # -------------------------------------------------------
    # Step 1: Load tags
    # -------------------------------------------------------
    print("\n[1/6] Loading tags...")
    tags_df = pd.read_csv("data/raw/steam-insights/tags.csv", on_bad_lines='skip')
    print(f"  Loaded {len(tags_df)} tag rows for {tags_df['app_id'].nunique()} unique games")
    
    tags_grouped = (
        tags_df
        .groupby('app_id')['tag']
        .apply(lambda x: ', '.join(sorted(set(t.strip().lower() for t in x))))
        .reset_index()
    )
    tags_grouped.columns = ['app_id', 'tags_string']
    tags_grouped['tag_count'] = tags_grouped['tags_string'].apply(lambda x: len(x.split(', ')))
    tags_grouped['app_id'] = tags_grouped['app_id'].astype(int)
    
    print(f"  Grouped into {len(tags_grouped)} games with tag strings")
    print(f"  Avg tags per game: {tags_grouped['tag_count'].mean():.1f}")
    
    # -------------------------------------------------------
    # Step 2: Parse promotional data
    # -------------------------------------------------------
    print("\n[2/6] Parsing promotional data...")
    promo_df = parse_promotional("data/raw/steam-insights/promotional.csv")
    promo_df['app_id'] = promo_df['app_id'].astype(int)
    print(f"  Games with 3+ screenshots: {len(promo_df)}")
    
    if len(promo_df) == 0:
        print("\n  FATAL: Could not extract any screenshot URLs.")
        return
    
    # -------------------------------------------------------
    # Step 3: Load game names
    # -------------------------------------------------------
    print("\n[3/6] Loading game names...")
    games_names = parse_games_names("data/raw/steam-insights/games.csv")
    print(f"  Loaded names for {len(games_names)} games")
    
    # -------------------------------------------------------
    # Step 4: Join tags + screenshots
    # -------------------------------------------------------
    print("\n[4/6] Joining tags with screenshot data...")
    master = pd.merge(tags_grouped, promo_df, on='app_id', how='inner')
    print(f"  Games with both tags AND 3+ screenshots: {len(master)}")
    
    master = pd.merge(master, games_names, on='app_id', how='left')
    master['name'] = master['name'].fillna('Unknown')
    
    named_count = (master['name'] != 'Unknown').sum()
    print(f"  Games with known names: {named_count} / {len(master)}")
    
    # -------------------------------------------------------
    # Step 5: Filter and clean
    # -------------------------------------------------------
    print("\n[5/6] Filtering and cleaning...")
    
    before = len(master)
    master = master[master['tag_count'] >= 3]
    print(f"  Removed {before - len(master)} games with < 3 tags")
    
    before = len(master)
    master = master.drop_duplicates(subset='app_id')
    print(f"  Removed {before - len(master)} duplicate app_ids")
    
    print(f"  Final dataset: {len(master)} games")
    
    if len(master) == 0:
        print("\n  FATAL: No games in final dataset.")
        return
    
    # -------------------------------------------------------
    # Step 6: Create train/val/test split
    # -------------------------------------------------------
    print("\n[6/6] Creating train/val/test split...")
    from sklearn.model_selection import train_test_split
    
    train_ids, temp_ids = train_test_split(
        master['app_id'].values,
        test_size=0.30,
        random_state=42
    )
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=42
    )
    
    master['split'] = 'train'
    master.loc[master['app_id'].isin(val_ids), 'split'] = 'val'
    master.loc[master['app_id'].isin(test_ids), 'split'] = 'test'
    
    print(f"  Train: {(master['split'] == 'train').sum()}")
    print(f"  Val:   {(master['split'] == 'val').sum()}")
    print(f"  Test:  {(master['split'] == 'test').sum()}")
    
    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_cols = ['app_id', 'name', 'tags_string', 'tag_count', 'url_0', 'url_1', 'url_2', 'screenshot_count', 'split']
    master[save_cols].to_csv(output_path / "games_master.csv", index=False)
    
    print(f"\n  Saved to {output_path / 'games_master.csv'}")
    
    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SAMPLE ROWS")
    print("=" * 60)
    for _, row in master[save_cols].head(5).iterrows():
        print(f"\n  app_id: {row['app_id']}")
        print(f"  name: {row['name']}")
        print(f"  tags: {row['tags_string'][:80]}...")
        print(f"  screenshots: {row['screenshot_count']}")
        print(f"  split: {row['split']}")
        print(f"  url_0: {row['url_0'][:90]}...")
    
    print(f"\n{'=' * 60}")
    print("TAG DISTRIBUTION IN FINAL DATASET")
    print("=" * 60)
    all_tags = master['tags_string'].str.split(', ').explode()
    print(all_tags.value_counts().head(20).to_string())
    
    print(f"\n{'=' * 60}")
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()