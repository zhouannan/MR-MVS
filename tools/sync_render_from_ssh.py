import os
import shutil
import subprocess
from pathlib import Path


HOST = "121.36.253.161"
PORT = "25213"
USER = "zan"
REMOTE_BASE = "/mnt/remote_data/DTU/render"
LOCAL_BASE = Path("static/render_compare")

SSH_EXE = r"C:\Windows\System32\OpenSSH\ssh.exe"
SCP_EXE = r"C:\Windows\System32\OpenSSH\scp.exe"
ASKPASS = str((Path(__file__).resolve().parent / "askpass.bat"))

METHODS = {
    "transmvsnet": "transmvsnet",
    "geomvsnet": "geomvsnet",
    "mvsformer": "mvsformerplusplus",
    "mrmvs_1": "mrmvs",
}


def auth_env():
    env = os.environ.copy()
    env["SSH_ASKPASS"] = ASKPASS
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env["DISPLAY"] = "1"
    return env


def run_cmd(cmd):
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=auth_env(),
        stdin=subprocess.DEVNULL,
    )


def list_scans():
    cmd = [
        SSH_EXE,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=NUL",
        "-p",
        PORT,
        f"{USER}@{HOST}",
        "ls",
        REMOTE_BASE,
    ]
    p = run_cmd(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Failed to list remote scans:\n{p.stderr}")
    scans = [x.strip() for x in p.stdout.splitlines() if x.strip().startswith("scan")]
    scans.sort()
    return scans


def copy_method(scan, remote_method, local_method):
    scan_dir = LOCAL_BASE / scan
    scan_dir.mkdir(parents=True, exist_ok=True)
    final_dir = scan_dir / local_method

    if final_dir.exists() and any(final_dir.iterdir()):
        print(f"[skip] {scan}/{local_method} already exists")
        return

    remote_path = f"{USER}@{HOST}:{REMOTE_BASE}/{scan}/{remote_method}"
    cmd = [
        SCP_EXE,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=NUL",
        "-P",
        PORT,
        "-r",
        remote_path,
        str(scan_dir),
    ]
    p = run_cmd(cmd)
    if p.returncode != 0:
        print(f"[miss] {scan}/{remote_method} ({p.returncode})")
        return

    copied_dir = scan_dir / remote_method
    if remote_method != local_method and copied_dir.exists():
        if final_dir.exists():
            shutil.rmtree(final_dir)
        copied_dir.rename(final_dir)
    print(f"[ok]   {scan}/{local_method}")


def main():
    LOCAL_BASE.mkdir(parents=True, exist_ok=True)
    scans = list_scans()
    print(f"Found {len(scans)} scans")
    for scan in scans:
        for remote_method, local_method in METHODS.items():
            copy_method(scan, remote_method, local_method)
    print("Done")


if __name__ == "__main__":
    main()
