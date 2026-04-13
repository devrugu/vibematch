"""
Debug: inspect the raw screenshot column to fix URL extraction.
Run from project root: python debug_screenshots.py
"""
import pandas as pd

promo = pd.read_csv("../data/raw/steam-insights/promotional.csv", on_bad_lines='skip')

# Show first 3 non-null screenshot entries, raw
print("=" * 60)
print("RAW SCREENSHOT COLUMN (first 3 non-null)")
print("=" * 60)

count = 0
for idx, row in promo.iterrows():
    val = row['screenshots']
    if pd.notna(val) and len(str(val)) > 10:
        print(f"\n--- app_id: {row['app_id']} ---")
        raw = str(val)
        # Print first 500 chars to see the actual format
        print(f"First 500 chars:\n{raw[:500]}")
        print(f"\nFull length: {len(raw)} chars")
        
        # Show all unique URL-like patterns
        import re
        # Try multiple patterns
        patterns = {
            'original (1920x1080)': r'https://shared\.akamai\.steamstatic\.com/store_item_assets/steam/apps/\d+/\d+\.1920x1080\.jpg\?t=\d+',
            'any steamstatic URL': r'https?://[^\s\"\\\}]+steamstatic[^\s\"\\\}]+\.jpg[^\s\"\\\}]*',
            'any https URL': r'https?://[^\s\"\\\}]+\.jpg[^\s\"\\\}]*',
            'path_full value': r'path_full[\"\\:\s]+([^\s\"\\\}]+)',
        }
        
        for name, pattern in patterns.items():
            matches = re.findall(pattern, raw)
            print(f"\nPattern '{name}': {len(matches)} matches")
            if matches:
                print(f"  First match: {matches[0][:120]}")
        
        count += 1
        if count >= 3:
            break

if count == 0:
    print("No non-null screenshot entries found!")
    print(f"\nScreenshots column null count: {promo['screenshots'].isna().sum()} / {len(promo)}")
    print(f"Sample values: {promo['screenshots'].head(10).tolist()}")