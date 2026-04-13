"""
VibeMatch — Step 3: Download Screenshots
Downloads 3 screenshots per game from Steam CDN.

These are direct static file URLs (not API calls), so there's no
strict rate limiting. We use async downloads with a concurrency limit
to be respectful and maximize throughput.

Run from project root: python src/data/download_screenshots.py

Options:
    --limit N       Only download first N games (for testing)
    --concurrency N Number of simultaneous downloads (default: 20)
    --resume        Skip games that already have all 3 images downloaded
"""
import pandas as pd
import aiohttp
import aiofiles
import asyncio
import os
import sys
import time
from pathlib import Path
from argparse import ArgumentParser


# Where to save images
IMAGE_DIR = Path("data/images")

# Track progress
stats = {
    'downloaded': 0,
    'skipped': 0,
    'failed': 0,
    'total': 0,
}


async def download_one(session, url, save_path, semaphore, retries=3):
    """Download a single image with retry logic."""
    async with semaphore:
        for attempt in range(retries):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        # Verify it's a real image (at least 5KB)
                        if len(content) < 5000:
                            stats['failed'] += 1
                            return False
                        # Save
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        async with aiofiles.open(save_path, 'wb') as f:
                            await f.write(content)
                        stats['downloaded'] += 1
                        return True
                    elif resp.status == 429:
                        # Rate limited — wait and retry
                        wait = 2 ** attempt
                        await asyncio.sleep(wait)
                        continue
                    else:
                        stats['failed'] += 1
                        return False
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < retries - 1:
                    await asyncio.sleep(1)
                    continue
                stats['failed'] += 1
                return False
    return False


async def download_game(session, row, semaphore):
    """Download all 3 screenshots for one game."""
    app_id = row['app_id']
    game_dir = IMAGE_DIR / str(app_id)
    
    # Check if already downloaded (for resume mode)
    existing = [
        (game_dir / f"screenshot_{i}.jpg").exists()
        for i in range(3)
    ]
    if all(existing):
        stats['skipped'] += 1
        return
    
    # Download each screenshot
    for i in range(3):
        url = row[f'url_{i}']
        save_path = game_dir / f"screenshot_{i}.jpg"
        
        if save_path.exists():
            stats['skipped'] += 1
            continue
        
        await download_one(session, url, save_path, semaphore)


async def main_async(df, concurrency):
    """Main async download loop."""
    semaphore = asyncio.Semaphore(concurrency)
    
    # Use a persistent session for connection pooling
    connector = aiohttp.TCPConnector(limit=concurrency, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Process in batches for progress reporting
        batch_size = 500
        total_games = len(df)
        
        for batch_start in range(0, total_games, batch_size):
            batch_end = min(batch_start + batch_size, total_games)
            batch = df.iloc[batch_start:batch_end]
            
            tasks = [
                download_game(session, row, semaphore)
                for _, row in batch.iterrows()
            ]
            
            await asyncio.gather(*tasks)
            
            # Progress report
            progress = batch_end / total_games * 100
            print(
                f"  [{progress:5.1f}%] "
                f"Games: {batch_end}/{total_games} | "
                f"Downloaded: {stats['downloaded']} | "
                f"Skipped: {stats['skipped']} | "
                f"Failed: {stats['failed']}"
            )


def main():
    parser = ArgumentParser(description="Download Steam game screenshots")
    parser.add_argument('--limit', type=int, default=0, help='Limit to N games (0 = all)')
    parser.add_argument('--concurrency', type=int, default=20, help='Simultaneous downloads')
    parser.add_argument('--resume', action='store_true', help='Skip already downloaded games')
    args = parser.parse_args()
    
    print("=" * 60)
    print("VibeMatch — Screenshot Downloader")
    print("=" * 60)
    
    # Load master CSV
    df = pd.read_csv("data/processed/games_master.csv")
    print(f"\nLoaded {len(df)} games from master CSV")
    
    if args.limit > 0:
        df = df.head(args.limit)
        print(f"Limited to first {args.limit} games")
    
    stats['total'] = len(df) * 3  # 3 screenshots per game
    
    print(f"Will download up to {stats['total']} images")
    print(f"Concurrency: {args.concurrency}")
    print(f"Saving to: {IMAGE_DIR.absolute()}")
    print()
    
    # Create image directory
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Start downloading
    start_time = time.time()
    asyncio.run(main_async(df, args.concurrency))
    elapsed = time.time() - start_time
    
    # Final report
    print(f"\n{'=' * 60}")
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Time: {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  Downloaded: {stats['downloaded']} images")
    print(f"  Skipped (already existed): {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
    
    # Verify: count how many games have all 3 screenshots
    complete = 0
    incomplete = 0
    for _, row in df.iterrows():
        game_dir = IMAGE_DIR / str(row['app_id'])
        has_all = all(
            (game_dir / f"screenshot_{i}.jpg").exists()
            for i in range(3)
        )
        if has_all:
            complete += 1
        else:
            incomplete += 1
    
    print(f"\n  Games with all 3 screenshots: {complete}")
    print(f"  Games with missing screenshots: {incomplete}")
    
    # Estimate disk usage
    total_size = sum(
        f.stat().st_size
        for f in IMAGE_DIR.rglob("*.jpg")
        if f.is_file()
    )
    print(f"  Total disk usage: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()