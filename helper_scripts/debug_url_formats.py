"""
Debug: understand what URL formats exist and why only 427 match.
Run from project root: python debug_url_formats.py
"""
import re

with open("../data/raw/steam-insights/promotional.csv", "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

print(f"Total data lines: {len(lines) - 1}")

# Categorize lines
has_1920 = 0
has_600 = 0
has_screenshots_field = 0
has_null_screenshots = 0
has_any_url = 0
total_checked = 0

# Count different screenshot URL patterns
for line in lines[1:]:
    total_checked += 1
    
    if '1920x1080' in line:
        has_1920 += 1
    if '600x338' in line:
        has_600 += 1
    if 'screenshot' in line.lower() or 'store_item_assets' in line:
        has_screenshots_field += 1
    if line.rstrip().endswith('\\N') or ',\\N,' in line:
        has_null_screenshots += 1
    if 'https://' in line:
        has_any_url += 1

print(f"\nURL FORMAT COUNTS:")
print(f"  Lines with 1920x1080 URLs: {has_1920}")
print(f"  Lines with 600x338 URLs: {has_600}")
print(f"  Lines with any store_item_assets URL: {has_screenshots_field}")
print(f"  Lines with any https URL: {has_any_url}")
print(f"  Lines with \\N (null fields): {has_null_screenshots}")

# Show a line that has 600x338 but NOT 1920x1080
print(f"\n{'=' * 60}")
print("SAMPLE: Line with 600x338 but NOT 1920x1080")
print("=" * 60)
count = 0
for line in lines[1:]:
    if '600x338' in line and '1920x1080' not in line:
        app_match = re.match(r'^"(\d+)"', line)
        app_id = app_match.group(1) if app_match else "?"
        print(f"\napp_id: {app_id}")
        # Extract all URLs
        urls = re.findall(r'https://[^\s"\\,}]+\.jpg[^\s"\\,}]*', line)
        print(f"  Total URLs found: {len(urls)}")
        for u in urls[:5]:
            print(f"    {u}")
        count += 1
        if count >= 3:
            break

# Show a line that HAS 1920x1080
print(f"\n{'=' * 60}")
print("SAMPLE: Line WITH 1920x1080")
print("=" * 60)
count = 0
for line in lines[1:]:
    if '1920x1080' in line:
        app_match = re.match(r'^"(\d+)"', line)
        app_id = app_match.group(1) if app_match else "?"
        # Count how many 1920x1080 URLs
        urls_1920 = re.findall(r'https://[^\s"\\,}]+1920x1080\.jpg[^\s"\\,}]*', line)
        urls_600 = re.findall(r'https://[^\s"\\,}]+600x338\.jpg[^\s"\\,}]*', line)
        print(f"\napp_id: {app_id} — {len(urls_1920)} full-res, {len(urls_600)} thumbnails")
        count += 1
        if count >= 3:
            break

# Show a line with NULL screenshots
print(f"\n{'=' * 60}")
print("SAMPLE: Lines with NULL screenshots")
print("=" * 60)
count = 0
for line in lines[1:]:
    if line.rstrip().endswith('\\N'):
        app_match = re.match(r'^"(\d+)"', line)
        app_id = app_match.group(1) if app_match else "?"
        print(f"  app_id: {app_id} — line length: {len(line)} chars — ends with: {repr(line[-30:])}")
        count += 1
        if count >= 5:
            break

# Now try extracting ALL screenshot URLs (both resolutions) and count games with 3+
print(f"\n{'=' * 60}")
print("GAMES WITH 3+ SCREENSHOTS (any resolution)")
print("=" * 60)
games_any_res = 0
games_3plus_any = 0
for line in lines[1:]:
    app_match = re.match(r'^"(\d+)"', line)
    if not app_match:
        continue
    # Find ALL screenshot URLs (any resolution)
    urls = re.findall(
        r'https://shared\.akamai\.steamstatic\.com/store_item_assets/steam/apps/\d+/\d+\.\d+x\d+\.jpg\?t=\d+',
        line
    )
    if urls:
        games_any_res += 1
    if len(urls) >= 3:
        games_3plus_any += 1

print(f"  Games with any screenshot URL: {games_any_res}")
print(f"  Games with 3+ screenshot URLs (any res): {games_3plus_any}")

# Also try just finding path_full URLs with a looser pattern
print(f"\n{'=' * 60}")
print("TRYING LOOSER URL PATTERN")
print("=" * 60)
games_loose = 0
for line in lines[1:]:
    app_match = re.match(r'^"(\d+)"', line)
    if not app_match:
        continue
    # Very loose: any steamstatic URL with .jpg
    urls = re.findall(r'https://shared\.akamai\.steamstatic\.com/[^\s"\\]+\.jpg', line)
    # Filter to just screenshot URLs (not header/background)
    screenshot_urls = [u for u in urls if '/header.' not in u and '/page_bg' not in u]
    if len(screenshot_urls) >= 3:
        games_loose += 1

print(f"  Games with 3+ screenshot URLs (loose pattern): {games_loose}")