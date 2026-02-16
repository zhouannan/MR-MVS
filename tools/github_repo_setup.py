from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


API = "https://api.github.com"


def req(method: str, url: str, token: str, payload: dict | None = None):
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    r = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "mr-mvs-publisher",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.getcode(), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"raw": body}
        return e.code, parsed


def main():
    token = os.environ.get("GH_TOKEN")
    owner = os.environ.get("GH_OWNER", "zhouannan")
    repo = os.environ.get("GH_REPO", "MR-MVS")
    private = os.environ.get("GH_PRIVATE", "false").lower() == "true"
    if not token:
        print("ERR: GH_TOKEN not set")
        sys.exit(2)

    code, body = req("GET", f"{API}/repos/{owner}/{repo}", token)
    if code == 200:
        print("exists", body.get("html_url", ""))
        return
    if code != 404:
        print("ERR GET", code, body)
        sys.exit(1)

    payload = {"name": repo, "private": private, "auto_init": False}
    code, body = req("POST", f"{API}/user/repos", token, payload)
    if code not in (200, 201):
        print("ERR CREATE", code, body)
        sys.exit(1)
    print("created", body.get("html_url", ""))


if __name__ == "__main__":
    main()

