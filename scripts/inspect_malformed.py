
import gzip

SOURCE_PATH = "data/raw/finefoods.txt"

examples = [
    "88 years old. ...",
    "...creative powers b...",
    "School Princi...",
    "I am a voracious reader/li...",
]

with open(SOURCE_PATH, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

for example in examples:
    matches = [
        i for i, line in enumerate(lines)
        if example in line
    ]

    print("\n" + "=" * 70)
    print("Searching for:", example)
    print("Matches:", len(matches))

    for i in matches[:3]:
        start = max(0, i - 8)
        end = min(len(lines), i + 9)

        print(f"\nContext around source line {i + 1}:")
        for j in range(start, end):
            print(f"{j + 1}: {lines[j].rstrip()}")

