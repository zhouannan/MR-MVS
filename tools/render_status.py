from pathlib import Path

root = Path("static/render_compare")
scans = sorted([p for p in root.iterdir() if p.is_dir()]) if root.exists() else []
print("scan_count", len(scans))
for s in scans:
    methods = sorted([m.name for m in s.iterdir() if m.is_dir()])
    print(s.name, ",".join(methods))
