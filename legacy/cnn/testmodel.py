import os
from pathlib import Path

root = Path("one_from_each")   # adjust path if needed
total = 0
per_class = {}

for d in sorted(root.iterdir()):
    if d.is_dir() and d.name != "Indian_TEST":   # skip test set
        n = sum(1 for f in d.iterdir() if f.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"])
        per_class[d.name] = n
        total += n

print("Total images (train pool):", total)
print("Classes:", len(per_class))
for k,v in per_class.items():
    print(f"{k}: {v}")
