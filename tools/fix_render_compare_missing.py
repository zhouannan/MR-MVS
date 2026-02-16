from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


HOST = "121.36.253.161"
PORT = "25213"
USER = "zan"
REMOTE_BASE = "/mnt/remote_data/DTU/render"

ROOT = Path("static/render_compare")
ASKPASS = str((Path(__file__).resolve().parent / "askpass.bat"))

SCP_EXE = r"C:\Windows\System32\OpenSSH\scp.exe"

METHOD_REMOTE_TO_LOCAL = {
    "transmvsnet": "transmvsnet",
    "geomvsnet": "geomvsnet",
    "mvsformer": "mvsformerplusplus",
    "mrmvs_1": "mrmvs",
}
LOCAL_TO_REMOTE = {v: k for k, v in METHOD_REMOTE_TO_LOCAL.items()}


def auth_env():
    env = os.environ.copy()
    env["SSH_ASKPASS"] = ASKPASS
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env["DISPLAY"] = "1"
    return env


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=auth_env(),
        stdin=subprocess.DEVNULL,
    )


def scp_dir(remote_path: str, local_dir: Path) -> bool:
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        SCP_EXE,
        "-C",
        "-o",
        "ConnectTimeout=30",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=NUL",
        "-P",
        PORT,
        "-r",
        remote_path,
        str(local_dir),
    ]
    p = run(cmd)
    if p.returncode != 0:
        print(f"[scp_dir fail] {remote_path} -> {local_dir} ({p.returncode})")
        if p.stderr.strip():
            print(p.stderr.strip())
        return False
    return True


def scp_file(remote_path: str, local_path: Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        SCP_EXE,
        "-C",
        "-o",
        "ConnectTimeout=30",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=NUL",
        "-P",
        PORT,
        remote_path,
        str(local_path),
    ]
    p = run(cmd)
    if p.returncode != 0:
        print(f"[scp_file fail] {remote_path} -> {local_path} ({p.returncode})")
        if p.stderr.strip():
            print(p.stderr.strip())
        return False
    return True


def ensure_method_dir(scan: str, local_method: str) -> Path:
    return ROOT / scan / local_method


def download_method_dir(scan: str, local_method: str) -> bool:
    remote_method = LOCAL_TO_REMOTE[local_method]
    scan_dir = ROOT / scan
    scan_dir.mkdir(parents=True, exist_ok=True)

    remote = f"{USER}@{HOST}:{REMOTE_BASE}/{scan}/{remote_method}"
    ok = scp_dir(remote, scan_dir)
    if not ok:
        return False

    copied = scan_dir / remote_method
    target = scan_dir / local_method
    if remote_method != local_method:
        if target.exists():
            shutil.rmtree(target)
        if copied.exists():
            copied.rename(target)
    return True


def fix_missing_files(scan: str, local_method: str, missing: list[str]) -> None:
    # If too many missing, redownload whole dir.
    if len(missing) > 10:
        print(f"[redownload] {scan}/{local_method} missing={len(missing)}")
        d = ensure_method_dir(scan, local_method)
        if d.exists():
            shutil.rmtree(d)
        download_method_dir(scan, local_method)
        return

    remote_method = LOCAL_TO_REMOTE[local_method]
    for fn in missing:
        remote = f"{USER}@{HOST}:{REMOTE_BASE}/{scan}/{remote_method}/{fn}"
        local = ensure_method_dir(scan, local_method) / fn
        print(f"[get] {scan}/{local_method}/{fn}")
        scp_file(remote, local)


def main():
    audit_path = ROOT / "audit_missing.json"
    if not audit_path.exists():
        raise SystemExit("ERR: run tools/audit_render_compare.py first")
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    scans = report.get("scans", {})
    if not scans:
        print("Nothing missing.")
        return

    for scan, r in scans.items():
        # Missing whole method directories
        for m in r.get("missing_methods", []):
            print(f"[download] {scan}/{m}")
            download_method_dir(scan, m)

        # Missing specific files
        for m, v in r.get("methods", {}).items():
            missing = v.get("missing") or []
            if missing:
                fix_missing_files(scan, m, missing)

    print("Fix pass done.")


if __name__ == "__main__":
    main()

