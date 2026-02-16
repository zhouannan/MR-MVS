from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("static/render_compare")
METHODS = ["transmvsnet", "geomvsnet", "mvsformerplusplus", "mrmvs"]
EXPECTED = [f"{i:08d}.jpg" for i in range(49)]


def audit_scan(scan_dir: Path) -> dict:
    out = {"missing_methods": [], "methods": {}}
    for m in METHODS:
        mdir = scan_dir / m
        if not mdir.is_dir():
            out["missing_methods"].append(m)
            continue
        present = {p.name for p in mdir.iterdir() if p.is_file()}
        missing = [name for name in EXPECTED if name not in present]
        # Extra files are fine (ignored), we only care expected 49.
        out["methods"][m] = {
            "present_expected": 49 - len(missing),
            "missing": missing,
        }
    return out


def main():
    report = {"root": str(ROOT.as_posix()), "expected_count": 49, "expected_names": EXPECTED, "scans": {}}
    if not ROOT.exists():
        print("ERR: static/render_compare does not exist")
        return

    scans = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("scan")])
    for scan in scans:
        r = audit_scan(scan)
        incomplete = bool(r["missing_methods"]) or any(v["missing"] for v in r["methods"].values())
        if incomplete:
            report["scans"][scan.name] = r

    (ROOT / "audit_missing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"scans_total={len(scans)} incomplete={len(report['scans'])}")
    if report["scans"]:
        for s, r in report["scans"].items():
            mm = ",".join(r["missing_methods"]) if r["missing_methods"] else "-"
            print(f"{s}: missing_methods={mm}")
            for m, v in r["methods"].items():
                if v["missing"]:
                    print(f"  {m}: missing_files={len(v['missing'])} present_expected={v['present_expected']}")


if __name__ == "__main__":
    main()
