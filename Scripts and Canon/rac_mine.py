import argparse, os, sys, json, re, time, logging, hashlib, statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple, Iterable, Set
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dateutil import parser as dtparse
from github import Github, GithubException, RateLimitExceededException, BadCredentialsException
from github import Auth

RUN_TS = datetime.now(timezone.utc).isoformat()

HTTP = requests.Session()
HTTP.timeout = 12
HTTP.headers.update({"Accept": "application/vnd.github+json", "User-Agent": "rac-mine-rules/2.1"})

logger = logging.getLogger("rac")
logger.setLevel(logging.INFO)
_stream = logging.StreamHandler(sys.stdout)
_stream.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_stream)

def setup_file_logging(path: Optional[str]):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fh = logging.FileHandler(path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

@lru_cache(maxsize=1)
def ASOF() -> datetime:
    v = os.getenv("RAC_ASOF_UTC", "2025-09-01T00:00:00Z")
    if v.endswith("Z"): v = v.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(v).astimezone(timezone.utc)
    except:
        return datetime(2025, 9, 1, tzinfo=timezone.utc)

@lru_cache(maxsize=1)
def WIN_START() -> datetime:
    v = os.getenv("RAC_WIN_START_UTC", "2024-09-01T00:00:00Z")
    if v.endswith("Z"): v = v.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(v).astimezone(timezone.utc)
    except:
        return datetime(2024, 9, 1, tzinfo=timezone.utc)

BOT_TAILS = ("[bot]", "-bot", "_bot")
BOT_NAME_PAT = re.compile(
    r"(bot$|^bot-|/bot$|dependabot|renovate|mergify|snyk|codecov|pre-commit|semantic-?release|"
    r"release-please|allcontributors|sonar|readthedocs|gha-?user|github-actions|action|automerge|"
    r"appveyor|travis|circleci|bors|bors-?)", re.I)

def looks_bot_login(login: Optional[str]) -> bool:
    """RABBIT-style bot detection via login patterns (no external list)"""
    if not login: return False
    l = login.lower()
    if BOT_NAME_PAT.search(l): return True
    if any(l.endswith(t) for t in BOT_TAILS): return True
    return False

def _with_backoff(fn, *args, **kwargs):
    """Exponential backoff wrapper for rate-limit and transient errors"""
    backoff = 5.0
    while True:
        try:
            return fn(*args, **kwargs)
        except RateLimitExceededException:
            logger.info(f"[rate-limit] sleeping {backoff:.1f}s")
            time.sleep(backoff); backoff = min(backoff*1.7, 90)
        except GithubException as e:
            if getattr(e, "status", None) in (403, 429):
                logger.info(f"[gh-{e.status}] sleeping {backoff:.1f}s")
                time.sleep(backoff); backoff = min(backoff*1.7, 90); continue
            raise
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code in (403, 429, 502, 503, 504):
                logger.info(f"[http {code}] sleeping {backoff:.1f}s")
                time.sleep(backoff); backoff = min(backoff*1.7, 60); continue
            raise
        except Exception:
            logger.info(f"[transient] sleeping {backoff:.1f}s")
            time.sleep(backoff); backoff = min(backoff*1.7, 60)

@lru_cache(maxsize=1)
def gh_client() -> Github:
    tok = os.getenv("GITHUB_TOKEN")
    if not tok:
        logger.error("GITHUB_TOKEN not set."); sys.exit(1)
    HTTP.headers["Authorization"] = f"Bearer {tok}"
    try:
        auth = Auth.Token(tok)
        return Github(auth=auth, per_page=100)
    except BadCredentialsException:
        logger.error("Bad GITHUB_TOKEN."); sys.exit(1)

SENTENCE_SPLIT_PAT = re.compile(r"[.!?]+")
WORD_PAT = re.compile(r"\b[\w'-]+\b")
CODE_BLOCK_PAT = re.compile(r"```[\s\S]*?```")
INDENT_CODE_PAT = re.compile(r"\n {4,}\S")
HEADING_PAT = re.compile(r"^#{1,6}\s|\n#{1,6}\s")
PASSIVE_PAT = re.compile(r"\b(is|was|were|be|been|being|are|am)\s+\w+ed\b", re.I)
URL_PAT = re.compile(r"https?://[^\s)>\]]+")

SYLLABLE_CACHE = {}
VOWELS = set("aeiouy")

def count_syllables(word: str) -> int:
    if word in SYLLABLE_CACHE: return SYLLABLE_CACHE[word]
    w = word.lower(); syll=0; prev=False
    for ch in w:
        is_v = ch in VOWELS
        if is_v and not prev: syll += 1
        prev = is_v
    if w.endswith("e") and syll>1: syll -= 1
    SYLLABLE_CACHE[word] = max(1, syll)
    return SYLLABLE_CACHE[word]

def text_stats(text: str) -> Dict[str, Any]:
    """Flesch readability, sentence/word counts, code blocks, headings, passive voice"""
    sents = [s.strip() for s in SENTENCE_SPLIT_PAT.split(text) if s.strip()]
    words = WORD_PAT.findall(text)
    syll = sum(count_syllables(w) for w in words) or 1
    n_words = max(1, len(words)); n_sents = max(1, len(sents))
    asl = n_words / n_sents
    asw = syll / n_words
    fres = 206.835 - 1.015*asl - 84.6*asw
    return {
        "n_words": n_words,
        "n_sents": n_sents,
        "flesch_reading_ease": round(fres, 2),
        "avg_sentence_len": round(asl, 2),
        "code_block_count": len(CODE_BLOCK_PAT.findall(text)) + len(INDENT_CODE_PAT.findall(text)),
        "heading_count": len(HEADING_PAT.findall(text)),
        "passive_hits": len(PASSIVE_PAT.findall(text))
    }

def _check_url(url: str, timeout: float=5.0) -> bool:
    try:
        r = HTTP.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code < 400
    except Exception:
        return False

def broken_link_ratio(text: str, timeout: float=5.0, cap: int=20) -> float:
    """Fraction of HTTP(S) links returning 4xx/5xx or timing out"""
    urls = URL_PAT.findall(text)
    if not urls: return 0.0
    urls = urls[:cap]; bad = 0
    with ThreadPoolExecutor(max_workers=5) as ex:
        fut = {ex.submit(_check_url, u, timeout): u for u in urls}
        for f in as_completed(fut):
            try:
                if not f.result(): bad += 1
            except Exception:
                bad += 1
    return round(bad / max(1, len(urls)), 3)

COMPLETENESS_PATS = {
    "overview": re.compile(r"\b(overview|about|introduction)\b", re.I),
    "install": re.compile(r"\b(install|installation|setup)\b", re.I),
    "quickstart": re.compile(r"\b(quick start|quickstart|getting started)\b", re.I),
    "dev_setup": re.compile(r"\b(development setup|dev setup|local development|build from source)\b", re.I),
    "testing": re.compile(r"\b(test|testing|run tests)\b", re.I),
    "contrib": re.compile(r"\b(contributing|how to contribute|pull requests)\b", re.I),
    "coc": re.compile(r"\b(code of conduct)\b", re.I),
    "troubleshooting": re.compile(r"\b(troubleshooting|faq)\b", re.I),
    "release": re.compile(r"\b(release|publishing|versioning|semantic version)\b", re.I)
}

def completeness_check(big_text: str) -> Dict[str, bool]:
    """Presence of common onboarding sections via keyword patterns"""
    t = big_text.lower()
    return {k: bool(p.search(t)) for k, p in COMPLETENESS_PATS.items()}

def months_between(iso_str: Optional[str]) -> Optional[float]:
    """Months from timestamp to ASOF"""
    if not iso_str: return None
    try:
        t = dtparse.isoparse(iso_str).astimezone(timezone.utc)
        delta = ASOF() - t
        return round(delta.days / 30.44, 2)
    except Exception:
        return None

def rubric_rule_scores(doc_texts: Dict[str,str], repo_meta: Dict[str,Any]) -> Dict[str, Any]:
    """Rules-based R/A/C scoring (0..5 scale) from documentation"""
    combined = "\n\n".join(doc_texts.values()) if doc_texts else ""
    ts = text_stats(combined) if combined else {"flesch_reading_ease":0,"avg_sentence_len":0,"code_block_count":0,"heading_count":0,"passive_hits":0,"n_words":0,"n_sents":1}
    comp = completeness_check(combined) if combined else {}
    
    r = 2.5
    if ts["flesch_reading_ease"] >= 60: r += 1.0
    if ts["avg_sentence_len"] <= 18:    r += 0.5
    if ts["code_block_count"] >= 2:     r += 0.5
    r = max(0.0, min(5.0, r))
    
    broken = broken_link_ratio(combined) if combined else 1.0
    msp = months_between(repo_meta.get("pushed_at"))
    a = 2.5
    if msp is not None:
        if msp <= 3: a += 1.0
        elif msp <= 6: a += 0.5
    if broken <= 0.1: a += 0.5
    rel_age = months_between(repo_meta.get("latest_release_at"))
    if rel_age and rel_age > 18: a -= 0.5
    a = max(0.0, min(5.0, a))
    
    c = 1.5 + 0.4 * sum(1 for v in comp.values() if v)
    c = max(0.0, min(5.0, c))
    
    return {"R": round(r,2), "A": round(a,2), "C": round(c,2),
            "readability_feats": ts, "completeness_sections": comp, "broken_link_ratio": broken}

DOC_CANDIDATES = [
    "README.md","README.rst","README.txt",
    "CONTRIBUTING.md","docs/CONTRIBUTING.md",".github/CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",".github/CODE_OF_CONDUCT.md","docs/CODE_OF_CONDUCT.md"
]
WORKFLOWS_DIR = ".github/workflows"

def safe_get(repo, path) -> Optional[str]:
    try:
        file = repo.get_contents(path)
        if getattr(file, "type", "") == "file":
            return file.decoded_content.decode("utf-8", errors="ignore")
    except (GithubException, Exception):
        return None
    return None

def exists_path(repo, path) -> bool:
    try:
        obj = repo.get_contents(path)
        return obj is not None
    except Exception:
        return False

def list_dir(repo, path) -> List[str]:
    try:
        items = repo.get_contents(path)
        if isinstance(items, list):
            return [it.path for it in items if getattr(it, "type", "") == "file"]
    except Exception:
        pass
    return []

def grab_docs(repo) -> Dict[str, str]:
    """Fetch README, CONTRIBUTING, CODE_OF_CONDUCT, and docs/*.md (up to 24 total)"""
    out = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        fut = {ex.submit(safe_get, repo, p): p for p in DOC_CANDIDATES}
        for f in as_completed(fut):
            path = fut[f]
            try:
                txt = f.result()
                if txt: out[path] = txt
            except Exception:
                pass
    try:
        contents = repo.get_contents("docs")
        if isinstance(contents, list):
            for c in contents:
                if getattr(c,"type","")=="file" and c.name.lower().endswith(".md"):
                    if len(out) > 24: break
                    try:
                        out[f"docs/{c.name}"] = repo.get_contents(c.path).decoded_content.decode("utf-8","ignore")
                    except Exception:
                        pass
    except Exception:
        pass
    return out

def fetch_repo_blob(g: Github, full_name: str) -> Dict[str,Any]:
    """Mine repo metadata, docs, contributor counts (human vs bot-like)"""
    try:
        r = _with_backoff(g.get_repo, full_name)
    except Exception:
        return {"full_name": full_name, "error": "not_found", "run_ts": RUN_TS}

    docs = grab_docs(r)
    latest_release_at = None
    try:
        rel = _with_backoff(r.get_releases)
        for rel0 in rel:
            latest_release_at = rel0.created_at.replace(tzinfo=timezone.utc).isoformat()
            break
    except Exception:
        pass

    bot_like_contribs=0; human_contribs=0
    try:
        contributors = _with_backoff(r.get_contributors)
        for i, c in enumerate(contributors):
            if i >= 60: break
            login = (getattr(c, "login", "") or "")
            if looks_bot_login(login): bot_like_contribs += 1
            else: human_contribs += 1
    except Exception:
        pass

    return {
        "full_name": r.full_name, "html_url": r.html_url, "description": r.description,
        "language": r.language, "stars": r.stargazers_count, "forks": r.forks_count,
        "open_issues": r.open_issues_count,
        "owner_type": "Org" if ((r.owner.type or '').lower()=='organization') else "User",
        "created_at": r.created_at.replace(tzinfo=timezone.utc).isoformat() if r.created_at else None,
        "pushed_at": r.pushed_at.replace(tzinfo=timezone.utc).isoformat() if r.pushed_at else None,
        "default_branch": r.default_branch, "docs": docs, "latest_release_at": latest_release_at,
        "bot_like_contribs": bot_like_contribs, "human_contribs": human_contribs, "run_ts": RUN_TS
    }

def score_entry_rules(entry: Dict[str,Any]) -> Dict[str,Any]:
    """Score R/A/C (rules only) with rationale for auditability"""
    docs = entry.get("docs") or {}
    rule = rubric_rule_scores(docs, entry)
    R, A, C = rule["R"], rule["A"], rule["C"]
    score = round(0.4*R + 0.3*A + 0.3*C, 2)
    return {
        "full_name": entry.get("full_name"),
        "language": entry.get("language"),
        "owner_type": entry.get("owner_type"),
        "stars": entry.get("stars"),
        "pushed_at": entry.get("pushed_at"),
        "latest_release_at": entry.get("latest_release_at"),
        "bot_like_contribs": entry.get("bot_like_contribs"),
        "human_contribs": entry.get("human_contribs"),
        "R": R, "A": A, "C": C, "score": score,
        "readability_feats": rule["readability_feats"],
        "completeness_sections": rule["completeness_sections"],
        "broken_link_ratio": rule["broken_link_ratio"],
        "rationale": {
            "readability": f"Flesch={rule['readability_feats']['flesch_reading_ease']} "
                           f"ASL={rule['readability_feats']['avg_sentence_len']} "
                           f"code_blocks={rule['readability_feats']['code_block_count']}",
            "actuality": f"broken_link_ratio={rule['broken_link_ratio']}, "
                         f"months_since_push={months_between(entry.get('pushed_at'))}",
            "completeness": f"sections_present="
                            f"{sorted([k for k,v in (rule['completeness_sections'] or {}).items() if v])}"
        },
        "run_ts": RUN_TS
    }

def paginate_rest(url: str, params: Dict[str,Any]) -> Iterable[Dict[str,Any]]:
    """Generic REST API pagination with rate-limit handling"""
    page = 1
    while True:
        p = dict(params); p.update({"per_page": 100, "page": page})
        r = HTTP.get(url, params=p)
        if r.status_code in (403,429):
            reset = r.headers.get("X-RateLimit-Reset"); retry_after = r.headers.get("Retry-After")
            wait = 60
            if retry_after:
                try: wait = int(float(retry_after))
                except: pass
            elif reset and str(reset).isdigit():
                wait = max(10, int(reset) - int(time.time()))
            logger.info(f"[rest {r.status_code}] sleeping {wait}s for {url}")
            time.sleep(wait); continue
        r.raise_for_status()
        items = r.json()
        if not items: break
        for it in items: yield it
        if len(items) < 100: break
        page += 1; time.sleep(0.25)

def fetch_prs_rest(owner: str, repo: str, since_iso: str) -> Iterable[Dict[str,Any]]:
    """List PRs (all states) newest first; stop when updated_at < since"""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    for pr in paginate_rest(url, {"state":"all","sort":"updated","direction":"desc"}):
        if pr.get("updated_at","") < since_iso: return
        yield pr

def fetch_issue_comments(owner: str, repo: str, num: int) -> List[Dict[str,Any]]:
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{num}/comments"
    return list(paginate_rest(url, {}))

def fetch_review_comments(owner: str, repo: str, num: int) -> List[Dict[str,Any]]:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{num}/comments"
    return list(paginate_rest(url, {}))

MAINTAINER_ASSOC = {"OWNER","MEMBER","COLLABORATOR"}

def _iter_recent_prs(repo, since_dt: datetime):
    pulls = _with_backoff(repo.get_pulls, state="all", sort="created", direction="desc")
    for pr in pulls:
        created = getattr(pr, "created_at", None)
        if not created: continue
        created = created.replace(tzinfo=timezone.utc)
        if created < since_dt:
            break
        yield pr

def get_ever_mergers_with_first_time(repo, max_scan: int = 500) -> Dict[str, datetime]:
    """Map login → earliest PR merge time (from merged_by field)"""
    g = gh_client()
    first_merge: Dict[str, datetime] = {}
    try:
        q = f"repo:{repo.full_name} is:pr is:merged"
        srch = _with_backoff(g.search_issues, query=q, sort="updated", order="desc")
        for i, item in enumerate(srch):
            if i >= max_scan: break
            if not getattr(item, "pull_request", None): continue
            try:
                p = _with_backoff(repo.get_pull, item.number)
                mb = getattr(p, "merged_by", None)
                mt = getattr(p, "merged_at", None)
                if mb and mt and (getattr(mb, "type", "") or "").lower() == "user":
                    login = getattr(mb, "login", "") or ""
                    if login and (not looks_bot_login(login)):
                        t = mt.replace(tzinfo=timezone.utc)
                        if login not in first_merge or t < first_merge[login]:
                            first_merge[login] = t
            except Exception:
                pass
    except Exception:
        pass
    return first_merge

def is_human_user(user) -> bool:
    try:
        t = (getattr(user, "type", "") or "").lower()
        login = getattr(user, "login", "") or ""
        return t == "user" and (not looks_bot_login(login))
    except Exception:
        return False

def infer_maintainers(repo, *, since_days=365, max_commits=300, max_prs_scan=400, max_merged_prs_scan=500) -> Tuple[Set[str], Dict[str, datetime]]:
    """
    Maintainers = EVER merged ∪ (association signals last year) ∪ collaborators ∪ recent committers.
    Returns (maintainer_logins, ever_merge_first_time_map).
    """
    asof = ASOF()
    since = asof - timedelta(days=since_days)
    maintainers: Set[str] = set()

    def _add(login: Optional[str]):
        if login and (not looks_bot_login(login)):
            maintainers.add(login)

    # 0) people who have EVER merged a PR (+ earliest merge time)
    ever_map = get_ever_mergers_with_first_time(repo, max_scan=max_merged_prs_scan)
    for login in ever_map.keys():
        _add(login)

    # 1) association from recent PR interactions
    try:
        scanned = 0
        for pr in _iter_recent_prs(repo, since):
            scanned += 1
            if scanned > max_prs_scan: break
            try:
                for rv in pr.get_reviews():
                    u = getattr(rv, "user", None)
                    assoc = (getattr(rv, "author_association","") or "").upper()
                    if u and assoc in MAINTAINER_ASSOC and is_human_user(u):
                        _add(getattr(u, "login", ""))
            except Exception:
                pass
            try:
                ic = list(pr.get_issue_comments()); rc = list(pr.get_review_comments())
                for c in ic + rc:
                    u = getattr(c, "user", None)
                    assoc = (getattr(c, "author_association","") or "").upper()
                    if u and assoc in MAINTAINER_ASSOC and is_human_user(u):
                        _add(getattr(u, "login",""))
            except Exception:
                pass
    except Exception:
        pass

    # 2) recent default-branch committers
    try:
        default_branch = getattr(repo, "default_branch", None) or "main"
        commits = repo.get_commits(since=since, sha=default_branch)
        for i, c in enumerate(commits):
            if i >= max_commits: break
            gh_user = getattr(c, "author", None) or getattr(c, "committer", None)
            if gh_user and is_human_user(gh_user):
                _add(getattr(gh_user, "login", ""))
    except Exception:
        pass

    # 3) collaborators API (best-effort)
    try:
        for u in repo.get_collaborators():
            if is_human_user(u):
                _add(getattr(u, "login",""))
    except Exception:
        pass  # often 403

    logger.info(f"[maintainers] {repo.full_name}: inferred {len(maintainers)} maintainers (ever mergers included)")
    return maintainers, ever_map

def compute_comment_mentorship_metrics(repo, *, since_days=365) -> dict:
    """Mentorship proxies from PR comments (365d): first-reply latency, maintainer/author ratio, links/comment"""
    asof = ASOF()
    since = asof - timedelta(days=since_days)
    maintainers_set, _ = infer_maintainers(repo, since_days=since_days)

    pr_count = 0
    prs_with_maint_comment = 0
    first_reply_latencies_hrs = []

    maintainer_comment_count = 0
    author_comment_count = 0
    total_human_comments = 0
    total_link_count = 0

    comment_texts = []
    CHAR_CAP = 120_000
    PER_COMMENT_CAP = 1500

    pulls = _with_backoff(repo.get_pulls, state="all", sort="created", direction="desc")
    for pr in pulls:
        created = getattr(pr, "created_at", None)
        if not created: continue
        created = created.replace(tzinfo=timezone.utc)
        if created < since: break

        pr_count += 1
        if pr_count % 25 == 0:
            logger.info(f"[comments] {repo.full_name}: scanned {pr_count} PRs since {since.date()}")

        pr_author = (getattr(getattr(pr, "user", None), "login", "") or "")
        issue_comments, review_comments = [], []
        try: issue_comments = list(pr.get_issue_comments())
        except Exception: pass
        try: review_comments = list(pr.get_review_comments())
        except Exception: pass

        has_maintainer_comment = False
        first_maintainer_ts = None
        pr_author_comments = 0
        pr_maintainer_comments = 0

        for c in issue_comments + review_comments:
            user = getattr(c, "user", None)
            if not user or not is_human_user(user):
                continue
            login = getattr(user, "login", "") or ""
            assoc = (getattr(c, "author_association", "") or "").upper()
            ctime = getattr(c, "created_at", None)
            ctime = ctime.replace(tzinfo=timezone.utc) if ctime else None
            body = (getattr(c, "body", "") or "")[:PER_COMMENT_CAP]

            total_human_comments += 1
            total_link_count += len(URL_PAT.findall(body))
            if len("".join(comment_texts)) < CHAR_CAP:
                comment_texts.append(body)

            is_maint = (login in maintainers_set) or (assoc in MAINTAINER_ASSOC)
            if login == pr_author:
                pr_author_comments += 1
            elif is_maint:
                pr_maintainer_comments += 1
                has_maintainer_comment = True
                if created and ctime and ctime >= created:
                    if first_maintainer_ts is None or ctime < first_maintainer_ts:
                        first_maintainer_ts = ctime

        maintainer_comment_count += pr_maintainer_comments
        author_comment_count += pr_author_comments

        if has_maintainer_comment:
            prs_with_maint_comment += 1
        if first_maintainer_ts and created:
            dt_hrs = (first_maintainer_ts - created).total_seconds() / 3600.0
            first_reply_latencies_hrs.append(dt_hrs)

    median_latency = float(sorted(first_reply_latencies_hrs)[len(first_reply_latencies_hrs)//2]) if first_reply_latencies_hrs else None
    maint_to_author_ratio = None
    if maintainer_comment_count or author_comment_count:
        maint_to_author_ratio = round(maintainer_comment_count / max(1, author_comment_count), 3)
    links_per_comment = round(total_link_count / max(1, total_human_comments), 3) if total_human_comments else 0.0
    share_pr_with_maint = round(prs_with_maint_comment / max(1, pr_count), 3) if pr_count else 0.0

    convo_text = "\n\n".join(comment_texts)
    ts = text_stats(convo_text) if convo_text else {
        "n_words": 0, "n_sents": 1, "flesch_reading_ease": 0.0, "avg_sentence_len": 0.0,
        "code_block_count": 0, "heading_count": 0, "passive_hits": 0
    }
    convo_broken = broken_link_ratio(convo_text) if convo_text else 0.0
    R_convo = 2.5
    if ts["flesch_reading_ease"] >= 60: R_convo += 1.0
    if ts["avg_sentence_len"] <= 18:   R_convo += 0.5
    if ts["code_block_count"] >= 1:    R_convo += 0.3
    R_convo = max(0.0, min(5.0, R_convo))

    logger.info(
        f"[comments-done] {repo.full_name} PRs={pr_count} PRs≥1Maint={prs_with_maint_comment} "
        f"median_first_maint_reply_hrs={median_latency} maintainer_to_author_ratio={maint_to_author_ratio} "
        f"links_per_comment={links_per_comment} R_convo_rule_365={R_convo:.2f}"
    )

    return {
        "maintainer_first_reply_median_hrs_365": median_latency,
        "pr_with_human_maint_comment_rate_365": share_pr_with_maint,
        "maintainer_to_author_comment_ratio_365": maint_to_author_ratio,
        "links_per_comment_365": links_per_comment,
        "denom_prs_comment_window_365": pr_count,
        "count_prs_with_maint_comment_365": prs_with_maint_comment,
        "count_maintainer_comments_365": maintainer_comment_count,
        "count_author_comments_365": author_comment_count,
        "count_human_comments_365": total_human_comments,
        "count_links_365": total_link_count,
        "convo_readability_feats": ts,
        "convo_broken_link_ratio_365": convo_broken,
        "R_convo_rule_365": round(R_convo, 2),
    }

def _parse_iso(s: str) -> datetime:
    return dtparse.isoparse(s).astimezone(timezone.utc)

def _has_commit_before(repo, author_login: str, start_dt: datetime) -> bool:
    try:
        commits = repo.get_commits(author=author_login, until=start_dt - timedelta(seconds=1))
        for i, _ in enumerate(commits):
            if i >= 1: return True
    except Exception:
        pass
    return False

def newcomer_stats_right_censored(repo, *, start_iso: str, end_iso: str) -> Dict[str, Any]:
    """Strict newcomers (right-censored) + conversion 1m/3m/6m + retention"""
    start = _parse_iso(start_iso)
    end   = _parse_iso(end_iso)
    asof  = ASOF()

    newcomers: Dict[str, Dict[str, Any]] = {}
    total_prs_in_window = 0

    pulls = _with_backoff(repo.get_pulls, state="all", sort="created", direction="asc")
    for pr in pulls:
        c = getattr(pr, "created_at", None)
        if not c: continue
        c = c.replace(tzinfo=timezone.utc)
        if c < start: continue
        if c >= end:  break
        total_prs_in_window += 1
        author = getattr(getattr(pr, "user", None), "login", None)
        if not author or looks_bot_login(author):
            continue
        if _has_commit_before(repo, author, start):
            continue
        rec = newcomers.setdefault(author, {"first_pr": None, "pr_times": []})
        rec["pr_times"].append(c)
        if rec["first_pr"] is None or c < rec["first_pr"]:
            rec["first_pr"] = c

    _, ever_merge_map = infer_maintainers(repo)

    def additional_prs_within(pr_times: List[datetime], first: datetime, horizon_days: int) -> int:
        cutoff = first + timedelta(days=horizon_days)
        return sum(1 for t in pr_times if first < t <= cutoff)

    def has_commit_after(user_login: str, t_after: datetime) -> bool:
        try:
            commits = repo.get_commits(author=user_login, since=t_after)
            for i, _ in enumerate(commits):
                if i >= 1: return True
        except Exception:
            return False
        return False

    horizons = {
        "1m": 30,
        "3m": 90,
        "6m": 180
    }
    conv_hits = {k: 0 for k in horizons}
    conv_denom = {k: 0 for k in horizons}
    add_counts: Dict[str, List[int]] = {k: [] for k in horizons}

    ret_hits = {"1m":0,"3m":0,"6m":0}
    ret_denom = {"1m":0,"3m":0,"6m":0}

    became_merger_ever_hits = 0
    became_merger_ever_denom = 0

    for login, rec in newcomers.items():
        first = rec.get("first_pr")
        if not first: continue
        for tag, days in horizons.items():
            if first + timedelta(days=days) <= asof:
                conv_denom[tag] += 1
                n_add = additional_prs_within(rec["pr_times"], first, days)
                add_counts[tag].append(n_add)
                if n_add >= 1:
                    conv_hits[tag] += 1

        h1 = first + timedelta(days=30)
        h3 = first + timedelta(days=90)
        h6 = first + timedelta(days=180)
        if h1 <= asof:
            ret_denom["1m"] += 1
            if has_commit_after(login, h1): ret_hits["1m"] += 1
        if h3 <= asof:
            ret_denom["3m"] += 1
            if has_commit_after(login, h3): ret_hits["3m"] += 1
        if h6 <= asof:
            ret_denom["6m"] += 1
            if has_commit_after(login, h6): ret_hits["6m"] += 1

        t_merge = ever_merge_map.get(login)
        if t_merge is not None:
            became_merger_ever_denom += 1
            if t_merge >= first:
                became_merger_ever_hits += 1

    def agg_stats(lst: List[int]) -> Dict[str, Optional[float]]:
        if not lst: return {"sum":0, "avg":None, "median":None}
        return {
            "sum": int(sum(lst)),
            "avg": round(sum(lst)/len(lst), 3),
            "median": float(statistics.median(lst))
        }

    out = {
        "newcomers_strict_window": len(newcomers),
        "total_prs_in_window": total_prs_in_window,
        "conversion_1m": round(conv_hits["1m"] / max(1, conv_denom["1m"]), 3) if conv_denom["1m"] else None,
        "denom_conv_1m": conv_denom["1m"],
        "conversion_3m": round(conv_hits["3m"] / max(1, conv_denom["3m"]), 3) if conv_denom["3m"] else None,
        "denom_conv_3m": conv_denom["3m"],
        "conversion_6m": round(conv_hits["6m"] / max(1, conv_denom["6m"]), 3) if conv_denom["6m"] else None,
        "denom_conv_6m": conv_denom["6m"],
        "additional_prs_1m": agg_stats(add_counts["1m"]),
        "additional_prs_3m": agg_stats(add_counts["3m"]),
        "additional_prs_6m": agg_stats(add_counts["6m"]),
        "retention_1m": round(ret_hits["1m"]/max(1,ret_denom["1m"]),3) if ret_denom["1m"] else None,
        "denom_ret_1m": ret_denom["1m"],
        "retention_3m": round(ret_hits["3m"]/max(1,ret_denom["3m"]),3) if ret_denom["3m"] else None,
        "denom_ret_3m": ret_denom["3m"],
        "retention_6m": round(ret_hits["6m"]/max(1,ret_denom["6m"]),3) if ret_denom["6m"] else None,
        "denom_ret_6m": ret_denom["6m"],
        "newcomer_became_merger_ever_rate": round(became_merger_ever_hits / max(1, became_merger_ever_denom), 3) if became_merger_ever_denom else None,
        "newcomer_became_merger_ever_count": became_merger_ever_hits,
        "denom_newcomer_became_merger_ever": became_merger_ever_denom
    }

    logger.info(
        "[newcomers] %s strict=%d conv(1m/3m/6m)=%s/%s/%s denoms=%s/%s/%s retention denoms=%s/%s/%s became_merger_ever=%s/%s",
        repo.full_name, len(newcomers),
        out["conversion_1m"], out["conversion_3m"], out["conversion_6m"],
        out["denom_conv_1m"], out["denom_conv_3m"], out["denom_conv_6m"],
        out["denom_ret_1m"], out["denom_ret_3m"], out["denom_ret_6m"],
        out["newcomer_became_merger_ever_count"], out["denom_newcomer_became_merger_ever"]
    )
    return out

def pr_merge_rate_90d_right_censored(repo, *, censor_days: int = 7) -> Dict[str, Any]:
    """
    Denominator: PRs created in [ASOF-90d, ASOF-censor_days]
    Numerator  : those PRs whose merged flag is True by ASOF
    """
    asof = ASOF()
    start = asof - timedelta(days=90)
    end   = asof - timedelta(days=max(0, int(censor_days)))

    if end <= start:
        return {"pr_merge_rate_90": None, "denom_prs_90": 0, "censor_days": censor_days}

    g = gh_client()
    prs_90 = 0; prs_merged_90 = 0
    try:
        q = f'repo:{repo.full_name} is:pr created:>={start.date()}..{end.date()}'
        for i, pr_issue in enumerate(_with_backoff(g.search_issues, query=q, sort="created", order="desc")):
            if i >= 500: break
            if not getattr(pr_issue, "pull_request", None): continue
            prs_90 += 1
            try:
                p = _with_backoff(repo.get_pull, pr_issue.number)
                if getattr(p, "merged", False): prs_merged_90 += 1
            except Exception:
                pass
    except Exception:
        pass

    rate = round(prs_merged_90 / max(1, prs_90), 3) if prs_90 else None
    logger.info("[pr-merge-90] %s prs=%d merged=%d rate=%s (censor_days=%d)",
                repo.full_name, prs_90, prs_merged_90, rate, censor_days)
    return {"pr_merge_rate_90": rate, "denom_prs_90": prs_90, "censor_days": censor_days}

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_shard_manifest(dump_root: str, shard_id: str, record: Dict[str,Any]):
    path = os.path.join(dump_root, f"{shard_id}.comments.index.json")
    arr = []
    if os.path.exists(path):
        try: arr = json.load(open(path,"r",encoding="utf-8"))
        except Exception: arr = []
    arr.append(record)
    write_json(path, arr)

def dump_repo_comments_365d(full_name: str, dump_root: str, shard_id: str) -> Dict[str, Any]:
    """
    Dumps last 365d PR comments to <dump_root>/<shard>/<owner>__<repo>/pr_<N>.comments.json
    Returns per-repo index summary for convenience.
    """
    owner, repo = full_name.split("/",1)
    asof = ASOF()
    since = (asof - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
    repo_dir = os.path.join(dump_root, shard_id, f"{owner}__{repo}")
    ensure_dir(repo_dir)

    pr_files = []
    totals = {"issue": 0, "review": 0, "total": 0}
    pr_count = 0

    logger.info(f"[comments] dumping PR comments since {since} for {full_name}")
    try:
        for pr in fetch_prs_rest(owner, repo, since):
            try: num = int(pr.get("number"))
            except: continue
            ic = fetch_issue_comments(owner, repo, num)
            rc = fetch_review_comments(owner, repo, num)

            pr_count += 1
            totals["issue"] += len(ic); totals["review"] += len(rc); totals["total"] += (len(ic)+len(rc))

            out = {
                "shard_id": shard_id,
                "repo": full_name,
                "pr_number": num,
                "asof": ASOF().isoformat(),
                "window": {"since": since, "until": ASOF().strftime("%Y-%m-%dT%H:%M:%SZ")},
                "counts": {"issue": len(ic), "review": len(rc), "total": len(ic)+len(rc)},
                "comments": []
            }
            def pack(c: Dict[str,Any], typ: str) -> Dict[str,Any]:
                user = c.get("user") or {}
                login = user.get("login","")
                return {
                    "id": c.get("id"),
                    "type": typ,
                    "created_at": c.get("created_at"),
                    "updated_at": c.get("updated_at"),
                    "author_association": c.get("author_association"),
                    "user": {"login": login, "type": (user.get("type") or ""), "looks_bot": looks_bot_login(login)},
                    "html_url": c.get("html_url"),
                    "body": (c.get("body") or "")
                }
            out["comments"].extend(pack(x,"issue") for x in ic)
            out["comments"].extend(pack(x,"review") for x in rc)

            pr_path = os.path.join(repo_dir, f"pr_{num}.comments.json")
            write_json(pr_path, out)
            pr_files.append(os.path.basename(pr_path))

    except Exception as e:
        logger.warning(f"[comments] failed for {full_name}: {e}")

    repo_index = {
        "shard_id": shard_id,
        "repo": full_name,
        "asof": ASOF().isoformat(),
        "counts": totals,
        "pr_count_dumped": pr_count,
        "pr_files": pr_files
    }
    write_json(os.path.join(repo_dir, "index.json"), repo_index)
    append_shard_manifest(dump_root, shard_id, {"repo": full_name, "index": os.path.join(f"{owner}__{repo}", "index.json")})
    return repo_index

def repo_onboarding_and_issue_features(repo) -> Dict[str,Any]:
    """Onboarding surfaces (README, docs, templates, workflows, discussions)"""
    doc_paths = {
        "README.md": "README.md", "README.rst": "README.rst", "README.txt": "README.txt",
        "CONTRIBUTING.md": "CONTRIBUTING.md", "docs/CONTRIBUTING.md": "docs/CONTRIBUTING.md",
        ".github/CONTRIBUTING.md": ".github/CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md": "CODE_OF_CONDUCT.md", ".github/CODE_OF_CONDUCT.md": ".github/CODE_OF_CONDUCT.md",
        "docs/CODE_OF_CONDUCT.md": "docs/CODE_OF_CONDUCT.md"
    }
    docs = {}; readme_words = 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        fut = {ex.submit(safe_get, repo, p): p for p in doc_paths}
        for f in as_completed(fut):
            p = fut[f]
            try:
                content = f.result()
                docs[p] = bool(content)
                if p.startswith("README") and content:
                    readme_words = len(WORD_PAT.findall(content))
            except Exception:
                docs[p] = False

    docs_md_count = 0
    try:
        contents = repo.get_contents("docs")
        if isinstance(contents, list):
            docs_md_count = sum(1 for c in contents if getattr(c,"type","")=="file" and c.name.lower().endswith(".md"))
    except Exception:
        pass

    template_paths = [
        ".github/ISSUE_TEMPLATE.md",".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md","ISSUE_TEMPLATE.md",
        "PULL_REQUEST_TEMPLATE.md",".github/PULL_REQUEST_TEMPLATE.md",".github/PULL_REQUEST_TEMPLATE/pull_request_template.md",
        ".github/CODEOWNERS","MAINTAINERS.md","GOVERNANCE.md", WORKFLOWS_DIR
    ]
    exists = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut = {ex.submit(exists_path, repo, p): p for p in template_paths}
        for f in as_completed(fut):
            p = fut[f]
            try: exists[p] = f.result()
            except: exists[p] = False

    has_issue_template = any(exists.get(p, False) for p in [".github/ISSUE_TEMPLATE.md",".github/ISSUE_TEMPLATE/bug_report.md",".github/ISSUE_TEMPLATE/feature_request.md","ISSUE_TEMPLATE.md"])
    has_pr_template    = any(exists.get(p, False) for p in ["PULL_REQUEST_TEMPLATE.md",".github/PULL_REQUEST_TEMPLATE.md",".github/PULL_REQUEST_TEMPLATE/pull_request_template.md"])
    has_codeowners     = exists.get(".github/CODEOWNERS", False)
    has_maintainers    = exists.get("MAINTAINERS.md", False)
    has_governance     = exists.get("GOVERNANCE.md", False)
    has_workflows      = exists.get(WORKFLOWS_DIR, False)
    workflows_count    = len(list_dir(repo, WORKFLOWS_DIR)) if has_workflows else 0
    discussions_enabled= bool(getattr(repo, "has_discussions", False))

    onboarding_surfaces_count = sum([
        bool(readme_words>0),
        bool(docs_md_count>0),
        docs.get("CONTRIBUTING.md", False) or docs.get("docs/CONTRIBUTING.md", False) or docs.get(".github/CONTRIBUTING.md", False),
        docs.get("CODE_OF_CONDUCT.md", False) or docs.get(".github/CODE_OF_CONDUCT.md", False) or docs.get("docs/CODE_OF_CONDUCT.md", False),
        has_issue_template, has_pr_template, has_codeowners, has_maintainers, has_governance
    ])

    return {
        "readme_words": readme_words,
        "docs_md_count": docs_md_count,
        "has_contributing": bool(docs.get("CONTRIBUTING.md", False) or docs.get("docs/CONTRIBUTING.md", False) or docs.get(".github/CONTRIBUTING.md", False)),
        "has_coc": bool(docs.get("CODE_OF_CONDUCT.md", False) or docs.get(".github/CODE_OF_CONDUCT.md", False) or docs.get("docs/CODE_OF_CONDUCT.md", False)),
        "has_issue_template": has_issue_template,
        "has_pr_template": has_pr_template,
        "has_codeowners": has_codeowners,
        "has_maintainers": has_maintainers,
        "has_governance": has_governance,
        "has_workflows": has_workflows,
        "workflows_count": workflows_count,
        "discussions_enabled": discussions_enabled,
        "onboarding_surfaces_count": onboarding_surfaces_count
    }

def pr_merge_rate_and_features(repo, *, pr_censor_days: int) -> Dict[str, Any]:
    out = repo_onboarding_and_issue_features(repo)
    out.update(pr_merge_rate_90d_right_censored(repo, censor_days=pr_censor_days))
    return out

def load_repo_names_from_json(path: str) -> List[str]:
    data = json.load(open(path,"r",encoding="utf-8"))
    names=[]
    if isinstance(data, list):
        for el in data:
            if isinstance(el, str):
                names.append(el)
            elif isinstance(el, dict) and "full_name" in el:
                names.append(el["full_name"])
    else:
        raise ValueError("JSON must be an array of full_name strings or objects with full_name")
    return names

def write_json_array(path: str, rows: List[Dict[str,Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    logger.info(f"[save] {path} sha256={sha256_file(path)}")

def cmd_mine_from_json(args):
    g = gh_client()
    names = load_repo_names_from_json(args.json)
    logger.info(f"[mine_from_json] loaded {len(names)} names from {args.json}")
    out_rows=[]
    total = len(names)
    for i, name in enumerate(names, 1):
        logger.info(f"[mine {i}/{total}] {name}")
        row = fetch_repo_blob(g, name)
        out_rows.append(row)
        if args.sleep>0: time.sleep(args.sleep)
    write_json_array(args.out, out_rows)

def cmd_score(args):
    rows = json.load(open(args.inp,"r",encoding="utf-8"))
    logger.info(f"[score] {len(rows)} rows in (rules-only)")
    out=[]
    for i, entry in enumerate(rows, 1):
        if entry.get("error"):
            out.append(entry); continue
        rec = score_entry_rules(entry)
        out.append(rec)
        if i % 10 == 0:
            logger.info(f"[score] {i}/{len(rows)}")
    write_json_array(args.out, out)

def cmd_features(args):
    rows = json.load(open(args.inp,"r",encoding="utf-8"))
    logger.info(f"[features] {len(rows)} inputs  ASOF=%s  WIN_START=%s", ASOF().isoformat(), WIN_START().isoformat())
    out=[]
    total = len(rows)
    g = gh_client()
    for i, entry in enumerate(rows, 1):
        if entry.get("error"):
            out.append({"full_name": entry.get("full_name"), "error": entry.get("error")}); continue
        fn = entry["full_name"]
        logger.info(f"[features {i}/{total}] {fn}")
        try:
            repo = _with_backoff(g.get_repo, fn)
        except Exception:
            out.append({"full_name": fn, "error": "fetch_failed"}); continue

        dump_index = dump_repo_comments_365d(fn, args.dump_comments_dir, args.shard_id)

        base = pr_merge_rate_and_features(repo, pr_censor_days=args.pr_censor_days)

        comm = compute_comment_mentorship_metrics(repo, since_days=365)

        nc = newcomer_stats_right_censored(
            repo,
            start_iso=WIN_START().isoformat(),
            end_iso=ASOF().isoformat()
        )

        rec = {
            "full_name": fn,
            "asof": ASOF().isoformat(),
            "run_ts": RUN_TS,
            "comment_dump_index": {
                "repo_index": os.path.join(args.dump_comments_dir, args.shard_id, f"{fn.split('/')[0]}__{fn.split('/')[1]}", "index.json"),
                "shard_manifest": os.path.join(args.dump_comments_dir, f"{args.shard_id}.comments.index.json"),
                "counts": dump_index.get("counts", {}),
                "pr_count_dumped": dump_index.get("pr_count_dumped")
            }
        }
        rec.update(base); rec.update(comm); rec.update(nc)
        out.append(rec)

        if args.sleep>0: time.sleep(args.sleep)

    write_json_array(args.out, out)

def cmd_to_csv(args):
    scored = json.load(open(args.scored,"r",encoding="utf-8"))
    fields = [
        "full_name","language","owner_type","stars","pushed_at","latest_release_at",
        "bot_like_contribs","human_contribs","R","A","C","score",
        "readability_feats.flesch_reading_ease",
        "readability_feats.avg_sentence_len",
        "readability_feats.code_block_count",
        "broken_link_ratio"
    ]
    rows=[]
    for d in scored:
        if d.get("error"): continue
        rows.append({
            "full_name": d.get("full_name"),
            "language": d.get("language"),
            "owner_type": d.get("owner_type"),
            "stars": d.get("stars"),
            "pushed_at": d.get("pushed_at"),
            "latest_release_at": d.get("latest_release_at"),
            "bot_like_contribs": d.get("bot_like_contribs"),
            "human_contribs": d.get("human_contribs"),
            "R": d.get("R"), "A": d.get("A"), "C": d.get("C"), "score": d.get("score"),
            "readability_feats.flesch_reading_ease": (d.get("readability_feats") or {}).get("flesch_reading_ease"),
            "readability_feats.avg_sentence_len": (d.get("readability_feats") or {}).get("avg_sentence_len"),
            "readability_feats.code_block_count": (d.get("readability_feats") or {}).get("code_block_count"),
            "broken_link_ratio": d.get("broken_link_ratio"),
        })
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)
    logger.info(f"[table] wrote {args.out_csv} sha256={sha256_file(args.out_csv)}")

def main():
    p = argparse.ArgumentParser(description="Rules-only mining → scoring → features (+ ALWAYS dump comments) with right-censoring")
    p.add_argument("--log", default=None, help="log file path")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine_from_json", help="mine repos from a JSON array (strings or {full_name})")
    m.add_argument("--json", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--sleep", type=float, default=0.0)
    m.set_defaults(func=cmd_mine_from_json)

    s = sub.add_parser("score", help="score R/A/C (rules only) → JSON array")
    s.add_argument("--inp", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_score)

    ft = sub.add_parser("features", help="extract practices & outcomes (right-censored) and ALWAYS dump comments → JSON array")
    ft.add_argument("--inp", required=True, help="scored or raw JSON (needs full_name)")
    ft.add_argument("--out", required=True)
    ft.add_argument("--dump_comments_dir", required=True, help="root dir to dump PR comments (per-PR files)")
    ft.add_argument("--shard_id", required=True, help="logical shard id, e.g., shard_01")
    ft.add_argument("--pr_censor_days", type=int, default=7, help="right-censor fresh PRs younger than this many days")
    ft.add_argument("--sleep", type=float, default=0.0)
    ft.set_defaults(func=cmd_features)

    t = sub.add_parser("to_csv", help="flatten scored JSON to analysis CSV")
    t.add_argument("--scored", required=True)
    t.add_argument("--out_csv", required=True)
    t.set_defaults(func=cmd_to_csv)

    args = p.parse_args()
    setup_file_logging(args.log)
    logger.info(f"[start] CMD={args.cmd} CWD={os.getcwd()}")
    args.func(args)

if __name__ == "__main__":
    main()