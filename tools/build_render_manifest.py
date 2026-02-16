import json
from pathlib import Path


ROOT = Path("static/render_compare")
METHODS = ["transmvsnet", "geomvsnet", "mvsformerplusplus", "mrmvs"]


def list_images(scan: Path):
    # Use intersection of filenames across all methods so UI never points to missing files.
    sets = []
    for m in METHODS:
        p = scan / m
        if not p.is_dir():
            return []
        sets.append({f.name for f in p.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg"}})
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def main():
    out = {"methods": METHODS, "scans": {}}
    if not ROOT.exists():
        ROOT.mkdir(parents=True, exist_ok=True)
    for scan in sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("scan")]):
        imgs = list_images(scan)
        if imgs:
            out["scans"][scan.name] = imgs
    (ROOT / "manifest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {ROOT / 'manifest.json'} with {len(out['scans'])} scans")


if __name__ == "__main__":
    main()
