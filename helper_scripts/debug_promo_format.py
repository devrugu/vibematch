"""
Debug: understand the exact structure of promotional.csv records.
Run from project root: python debug_promo_format.py
"""
import re

with open("../data/raw/steam-insights/promotional.csv", "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
print(f"\nHeader: {lines[0][:100]}")

# Show lines 1-20 with their raw repr (first 150 chars each)
print("\n" + "=" * 60)
print("FIRST 30 LINES (raw repr, first 150 chars each)")
print("=" * 60)
for i in range(1, min(31, len(lines))):
    print(f"\nLine {i}: {repr(lines[i][:150])}")

# Count lines that start with a pure app_id pattern
# A real record start should be: "digits","http  (app_id followed by header_image URL)
print("\n" + "=" * 60)
print("RECORD START PATTERNS")
print("=" * 60)

# Pattern 1: line starts with "digits","http
p1 = sum(1 for l in lines[1:] if re.match(r'^"\d+","https?://', l))
print(f'Lines matching "digits","http...": {p1}')

# Pattern 2: line starts with "digits" (any)
p2 = sum(1 for l in lines[1:] if re.match(r'^"\d+"', l))
print(f'Lines matching "digits" (any): {p2}')

# Pattern 3: line starts with "digits","
p3 = sum(1 for l in lines[1:] if re.match(r'^"\d+",', l))
print(f'Lines matching "digits",: {p3}')

# Show a few examples of each type of line start
print("\n" + "=" * 60)
print("SAMPLE LINE STARTS (first 80 chars)")
print("=" * 60)

# Categorize lines
record_lines = []
continuation_lines = []
for i, l in enumerate(lines[1:], start=1):
    if re.match(r'^"\d+","https?://', l):
        record_lines.append((i, l))
    else:
        continuation_lines.append((i, l))

print(f"\nRecord start lines: {len(record_lines)}")
print("First 5:")
for idx, l in record_lines[:5]:
    print(f"  Line {idx}: {l[:120]}")

print(f"\nContinuation lines: {len(continuation_lines)}")
print("First 10:")
for idx, l in continuation_lines[:10]:
    print(f"  Line {idx}: {repr(l[:120])}")