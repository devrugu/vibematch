"""
Debug: read raw bytes of promotional.csv to understand its format.
Run from project root: python debug_raw_file.py
"""

# Read the first 3000 bytes raw to see exact format
with open("../data/raw/steam-insights/promotional.csv", "r", encoding="utf-8", errors="replace") as f:
    header = f.readline()
    print("HEADER LINE:")
    print(repr(header[:200]))
    print()
    
    # Read first data line
    # But the line might be very long, so read carefully
    first_line = f.readline()
    print(f"FIRST DATA LINE (length: {len(first_line)} chars):")
    print(repr(first_line[:500]))
    print("...")
    print(repr(first_line[-200:]))
    print()
    
    # Check what delimiter is used
    print("DELIMITER ANALYSIS:")
    print(f"  Tabs in header: {header.count(chr(9))}")
    print(f"  Commas in header: {header.count(',')}")
    print(f"  Pipes in header: {header.count('|')}")
    print(f"  Semicolons in header: {header.count(';')}")
    
    # Check if it's tab-separated
    if header.count(chr(9)) >= 3:
        print("\n  -> Likely TAB-separated!")
        parts = header.strip().split(chr(9))
        print(f"  Header columns: {parts}")
    elif header.count(',') >= 3:
        print("\n  -> Likely COMMA-separated")
    
    # Also check the first data line for tab vs comma
    print(f"\n  Tabs in first data line: {first_line.count(chr(9))}")
    print(f"  Commas in first data line: {first_line.count(',')}")