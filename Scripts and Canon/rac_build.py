import os
import sys
import time
import json
import csv
import math
import random
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

import requests

GITHUB_API = "https://api.github.com"
TOKEN = os.getenv("GITHUB_TOKEN", "").strip()

WINDOW_START = "2024-09-01"
WINDOW_END_EXCL = "2025-09-01"

STAR_BANDS = [
    ("50..500", "50..500"),
    ("500..1000", "500..1000"),
    (">=1000", ">=1000"),
]

TOTAL_TARGET = 100
OWNER_TYPES = ("User", "Organization")
SEARCH_PER_PAGE = 100
SEARCH_MAX_PAGES = 10
EXCLUDE_FORKS = False

session = requests.Session()
session.headers.update({
    "Accept": "application/vnd.github+json",
    "User-Agent": "repo-miner/1.0"
})
if TOKEN:
    session.headers["Authorization"] = f"Bearer {TOKEN}"


def _sleep_with_msg(seconds: float):
    seconds = max(0.0, seconds)
    if seconds > 0:
        print(f"[rate-limit/backoff] sleeping {seconds:.1f}s...", flush=True)
        time.sleep(seconds)


def gh_request(method: str, url: str, params: dict = None, retries: int = 3) -> requests.Response:
    """GitHub request with rate-limit & abuse-detection handling"""
    attempt = 0
    while True:
        attempt += 1
        resp = session.request(method, url, params=params)
        
        if resp.status_code in (429, 403):
            retry_after = resp.headers.get("Retry-After")
            remaining = resp.headers.get("X-RateLimit-Remaining")
            reset = resp.headers.get("X-RateLimit-Reset")
            if retry_after:
                try:
                    _sleep_with_msg(float(retry_after))
                    continue
                except Exception:
                    pass
            if remaining == "0" and reset:
                try:
                    reset_ts = int(reset)
                    now = int(time.time())
                    _sleep_with_msg(reset_ts - now + 2)
                    continue
                except Exception:
                    _sleep_with_msg(10)
                    continue
        if 200 <= resp.status_code < 300:
            return resp

        if attempt >= retries:
            return resp

        if 500 <= resp.status_code < 600:
            _sleep_with_msg(2 ** attempt)
            continue

        _sleep_with_msg(2)
        continue


def search_repositories(stars_qualifier: str) -> List[dict]:
    """Search repos in star band, paginated up to SEARCH_MAX_PAGES"""
    q_parts = [f"stars:{stars_qualifier}", "is:public"]
    if EXCLUDE_FORKS:
        q_parts.append("fork:false")
    q = " ".join(q_parts)

    results = []
    for page in range(1, SEARCH_MAX_PAGES + 1):
        params = {
            "q": q,
            "sort": "stars",
            "order": "desc",
            "per_page": SEARCH_PER_PAGE,
            "page": page
        }
        resp = gh_request("GET", f"{GITHUB_API}/search/repositories", params=params)
        if resp.status_code != 200:
            print(f"[warn] search {stars_qualifier} page {page} -> {resp.status_code} {resp.text[:200]}", flush=True)
            break
        data = resp.json()
        items = data.get("items", [])
        if not items:
            break
        results.extend(items)
        if len(items) < SEARCH_PER_PAGE:
            break
    return results


def repo_has_license(full_name: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Check for detected license via /repos/{full_name}/license endpoint"""
    resp = gh_request("GET", f"{GITHUB_API}/repos/{full_name}/license")
    if resp.status_code == 200:
        j = resp.json()
        lic = j.get("license") or {}
        return True, lic.get("key"), lic.get("name")
    return False, None, None


def repo_has_pr_in_window(full_name: str, start: str, end_excl: str) -> bool:
    """Check for at least one PR created in [start, end_excl) using Search Issues API"""
    q = f"repo:{full_name} is:pr is:public created:{start}..{end_excl}"
    params = {"q": q, "per_page": 1}
    resp = gh_request("GET", f"{GITHUB_API}/search/issues", params=params)
    if resp.status_code != 200:
        print(f"[warn] PR search failed for {full_name}: {resp.status_code} {resp.text[:160]}", flush=True)
        return False
    data = resp.json()
    return (data.get("total_count") or 0) > 0


def get_contributors_min_count(full_name: str, min_needed: int = 5) -> int:
    """Return unique contributor count from first page (up to min_needed)"""
    params = {"anon": "1", "per_page": max(5, min_needed)}
    resp = gh_request("GET", f"{GITHUB_API}/repos/{full_name}/contributors", params=params)
    if resp.status_code != 200:
        print(f"[warn] contributors fetch failed for {full_name}: {resp.status_code} {resp.text[:160]}", flush=True)
        return 0
    arr = resp.json() or []
    return len(arr)


def has_at_least_commits(full_name: str, min_commits: int = 50) -> Tuple[bool, int]:
    """Sum 'contributions' from top 100 contributors to estimate total commits"""
    total = 0
    page = 1
    per_page = 100
    while page <= 1:
        params = {"anon": "1", "per_page": per_page, "page": page}
        resp = gh_request("GET", f"{GITHUB_API}/repos/{full_name}/contributors", params=params)
        if resp.status_code != 200:
            print(f"[warn] contributors (for commits) failed for {full_name}: {resp.status_code} {resp.text[:160]}", flush=True)
            break
        arr = resp.json() or []
        for c in arr:
            total += int(c.get("contributions", 0) or 0)
            if total >= min_commits:
                return True, total
        if len(arr) < per_page:
            break
        page += 1

    return (total >= min_commits), total


def allocate_targets(total: int, bands: List[Tuple[str, str]], owner_types: Tuple[str, str]) -> Dict[Tuple[str, str], int]:
    """Distribute target count evenly across bands, then 50/50 across owner types"""
    per_band_base = total // len(bands)
    remainder = total % len(bands)

    band_targets = []
    for i, (label, _) in enumerate(bands):
        count = per_band_base + (1 if i < remainder else 0)
        band_targets.append((label, count))

    allocation: Dict[Tuple[str, str], int] = {}
    extra_to_user = True
    for label, band_count in band_targets:
        half = band_count // 2
        odd = band_count % 2
        for ot in owner_types:
            allocation[(label, ot)] = half
        if odd:
            allocation[(label, owner_types[0] if extra_to_user else owner_types[1])] += 1
            extra_to_user = not extra_to_user
    return allocation


def collect() -> List[Dict[str, Any]]:
    """Search, filter, and collect repos meeting all criteria"""
    allocation = allocate_targets(TOTAL_TARGET, STAR_BANDS, OWNER_TYPES)
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {k: [] for k in allocation}

    for band_label, stars_qual in STAR_BANDS:
        print(f"\n=== Searching band stars:{stars_qual} ===")
        candidates = search_repositories(stars_qual)
        print(f"[info] found {len(candidates)} candidates in band {band_label}")

        random.shuffle(candidates)

        for item in candidates:
            owner = item.get("owner") or {}
            owner_type = owner.get("type")
            if owner_type not in OWNER_TYPES:
                continue

            key = (band_label, owner_type)
            target_for_bucket = allocation[key]
            if len(buckets[key]) >= target_for_bucket:
                continue

            full_name = item.get("full_name")
            if not full_name:
                continue

            if EXCLUDE_FORKS and item.get("fork"):
                continue

            has_license, license_key, license_name = repo_has_license(full_name)
            if not has_license:
                continue

            if not repo_has_pr_in_window(full_name, WINDOW_START, WINDOW_END_EXCL):
                continue

            contrib_count = get_contributors_min_count(full_name, min_needed=5)
            if contrib_count < 5:
                continue

            ok_commits, total_commits_est = has_at_least_commits(full_name, min_commits=50)
            if not ok_commits:
                continue

            row = {
                "full_name": full_name,
                "html_url": item.get("html_url"),
                "owner_type": owner_type,
                "stargazers_count": item.get("stargazers_count"),
                "license_key": license_key,
                "license_name": license_name,
                "has_license": True,
                "has_pr_created_in_window": True,
                "contributors_count_at_least": contrib_count,
                "total_commits_estimate": total_commits_est,
                "default_branch": item.get("default_branch"),
                "is_fork": bool(item.get("fork")),
                "forks_count": item.get("forks_count"),
                "open_issues_count": item.get("open_issues_count"),
                "pushed_at": item.get("pushed_at"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "stars_band": band_label,
            }

            buckets[key].append(row)

            all_full = all(len(buckets[k]) >= allocation[k] for k in buckets)
            if all_full:
                break

        all_full = all(len(buckets[k]) >= allocation[k] for k in buckets)
        if all_full:
            break

    ordered_rows: List[Dict[str, Any]] = []
    for band_label, _ in STAR_BANDS:
        for ot in OWNER_TYPES:
            ordered_rows.extend(buckets[(band_label, ot)])

    total_found = len(ordered_rows)
    print(f"\n[summary] Target: {TOTAL_TARGET} | Collected: {total_found}")
    for band_label, _ in STAR_BANDS:
        for ot in OWNER_TYPES:
            print(f"  - {band_label} / {ot}: {len(buckets[(band_label, ot)])} (target {allocation[(band_label, ot)]})")

    return ordered_rows


def write_outputs(rows: List[Dict[str, Any]]):
    """Write collected repos to timestamped CSV and JSON files"""
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    csv_path = f"repos_{ts}.csv"
    json_path = f"repos_{ts}.json"

    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "full_name","html_url","owner_type","stargazers_count","license_key","license_name",
            "has_license","has_pr_created_in_window","contributors_count_at_least","total_commits_estimate",
            "default_branch","is_fork","forks_count","open_issues_count","pushed_at","created_at","updated_at",
            "stars_band"
        ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\n[done] Wrote {len(rows)} rows")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")


def main():
    if not TOKEN:
        print("[note] No GITHUB_TOKEN found. You will likely hit strict rate limits. Export GITHUB_TOKEN to speed things up.", file=sys.stderr)

    rows = collect()
    write_outputs(rows)


if __name__ == "__main__":
    main()