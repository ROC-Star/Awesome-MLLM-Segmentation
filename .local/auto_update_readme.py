#!/usr/bin/env python3
import datetime as dt
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_API = "http://export.arxiv.org/api/query"
START = "<!-- AUTO_MLLM_IMAGE_SEG_START -->"
END = "<!-- AUTO_MLLM_IMAGE_SEG_END -->"
KEYWORDS = [
    "multimodal large language model image segmentation",
    "mllm reasoning image segmentation",
    "referring image segmentation multimodal large language model",
    "open-vocabulary image segmentation vision language model",
]
MLLM_HINTS = [
    "mllm",
    "multimodal large language model",
    "large multimodal model",
    "vision-language model",
    "vision language model",
    "vlm",
    "llava",
    "qwen-vl",
]
SEG_HINTS = [
    "image segmentation",
    "referring expression segmentation",
    "reasoning segmentation",
    "open-vocabulary semantic segmentation",
    "pixel grounding",
    "mask",
]

DEFAULT_SKILL = """
You are a strict literature curation assistant for MLLM segmentation.

Task:
- Keep only papers explicitly about multimodal LLM / LVLM / VLM + segmentation.
- Prioritize image segmentation (referring/reasoning/open-vocabulary).
- Include video segmentation only if strongly MLLM-centered.
- Prefer newer papers and avoid duplicates.
- Keep arXiv links canonical and include code link only when confident.

Output JSON only:
{
    "entries": [
        {
            "short_name": "string",
            "venue": "string",
            "title": "string",
            "arxiv_url": "https://arxiv.org/abs/<id>",
            "code_url": "https://... or empty",
            "rationale": "short sentence",
            "experiment_evidence": "short cue"
        }
    ]
}
""".strip()


def fetch(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def query_arxiv(keyword: str, max_results: int = 35):
    params = {
        "search_query": f"all:{keyword}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = ARXIV_API + "?" + urllib.parse.urlencode(params)
    xml_text = fetch(url)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("atom:entry", ns):
        id_url = (e.findtext("atom:id", default="", namespaces=ns) or "").strip()
        if not id_url:
            continue
        arxiv_id = id_url.rsplit("/", 1)[-1]
        title = " ".join((e.findtext("atom:title", default="", namespaces=ns) or "").split())
        summary = " ".join((e.findtext("atom:summary", default="", namespaces=ns) or "").split())
        published = (e.findtext("atom:published", default="", namespaces=ns) or "").strip()
        out.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": summary,
                "published": published,
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
            }
        )
    return out


def strip_html(raw: str) -> str:
    raw = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.I)
    raw = re.sub(r"<style[\s\S]*?</style>", " ", raw, flags=re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def extract_urls(text: str):
    urls = re.findall(r"https?://[^\s)\]]+", text)
    return [u.rstrip(".,;") for u in urls]


def fetch_abs_metadata(arxiv_id: str):
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    html = fetch(abs_url)
    text = strip_html(html)
    code_url = ""
    for u in extract_urls(html):
        if "github.com" in u or "gitlab.com" in u:
            code_url = u
            break
    if not code_url:
        for u in extract_urls(html):
            if any(x in u.lower() for x in ["project", "page", "demo"]):
                code_url = u
                break
    return {
        "abs_url": abs_url,
        "abs_text": text,
        "code_url": code_url,
    }


def has_required_signals(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in MLLM_HINTS) and any(k in t for k in SEG_HINTS)


def slim_candidate(raw: dict, max_abs_chars: int) -> dict:
    abs_text = " ".join(str(raw.get("abs_text", "")).split())
    if len(abs_text) > max_abs_chars:
        abs_text = abs_text[:max_abs_chars] + " ..."
    return {
        "arxiv_id": raw.get("arxiv_id", ""),
        "title": raw.get("title", ""),
        "summary": raw.get("summary", ""),
        "published": raw.get("published", ""),
        "arxiv_url": raw.get("arxiv_url", ""),
        "code_url": raw.get("code_url", ""),
        "abs_text_excerpt": abs_text,
    }


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def parse_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}
        try:
            out = json.loads(m.group(0))
            return out if isinstance(out, dict) else {}
        except json.JSONDecodeError:
            return {}


def call_ollama_batch(skill: str, candidates: list, model: str, timeout_sec: int, retries: int):
    prompt = {
        "today": dt.date.today().isoformat(),
        "task": "Select only MLLM-based IMAGE segmentation papers. Output JSON only.",
        "skill": skill,
        "schema": {
            "entries": [
                {
                    "short_name": "string",
                    "venue": "string",
                    "title": "string",
                    "arxiv_url": "https://arxiv.org/abs/<id>",
                    "code_url": "string or empty",
                    "rationale": "string",
                    "experiment_evidence": "string",
                }
            ]
        },
        "constraints": [
            "MLLM is mandatory",
            "Image segmentation focused",
            "Do not include video-only papers",
            "No duplicates",
            "Use canonical arXiv abs URL",
        ],
        "candidates": candidates,
    }

    body = {
        "model": model,
        "prompt": json.dumps(prompt, ensure_ascii=False),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    req = urllib.request.Request(
        os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate"),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_err = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            resp_text = (out.get("response") or "").strip()
            data = parse_json_object(resp_text)
            entries = data.get("entries", []) if isinstance(data, dict) else []
            return entries if isinstance(entries, list) else []
        except (TimeoutError, socket.timeout, urllib.error.URLError, json.JSONDecodeError) as err:
            last_err = err
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"ollama batch failed after {retries} attempts: {last_err}")


def extract_openai_text(payload: dict) -> str:
    text = (payload.get("output_text") or "").strip()
    if text:
        return text
    pieces = []
    for item in payload.get("output", []):
        for c in item.get("content", []):
            if c.get("type") in {"output_text", "text"}:
                t = c.get("text")
                if t:
                    pieces.append(t)
    return "\n".join(pieces).strip()


def call_openai_batch(skill: str, candidates: list, model: str, timeout_sec: int, retries: int):
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    prompt = {
        "today": dt.date.today().isoformat(),
        "task": "Select only MLLM-based IMAGE segmentation papers. Output JSON only.",
        "skill": skill,
        "schema": {
            "entries": [
                {
                    "short_name": "string",
                    "venue": "string",
                    "title": "string",
                    "arxiv_url": "https://arxiv.org/abs/<id>",
                    "code_url": "string or empty",
                    "rationale": "string",
                    "experiment_evidence": "string",
                }
            ]
        },
        "constraints": [
            "MLLM is mandatory",
            "Image segmentation focused",
            "Do not include video-only papers",
            "No duplicates",
            "Use canonical arXiv abs URL",
        ],
        "candidates": candidates,
    }

    body = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": "You are a strict literature curation assistant. Return JSON only.",
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            },
        ],
    }
    req = urllib.request.Request(
        os.environ.get("OPENAI_URL", "https://api.openai.com/v1/responses"),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_err = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            resp_text = extract_openai_text(out)
            data = parse_json_object(resp_text)
            entries = data.get("entries", []) if isinstance(data, dict) else []
            return entries if isinstance(entries, list) else []
        except (TimeoutError, socket.timeout, urllib.error.URLError, json.JSONDecodeError) as err:
            last_err = err
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"openai batch failed after {retries} attempts: {last_err}")


def call_llm(skill: str, candidates: list):
    backend = os.environ.get("LLM_BACKEND", "auto").strip().lower()
    if backend not in {"auto", "ollama", "openai"}:
        backend = "auto"
    if backend == "auto":
        backend = "openai" if os.environ.get("OPENAI_API_KEY") else "ollama"

    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3.5:35b-a3b-nvfp4")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-5.3-codex")
    timeout_sec = int(os.environ.get("OLLAMA_TIMEOUT_SEC", "240"))
    retries = int(os.environ.get("OLLAMA_RETRIES", "2"))
    batch_size = int(os.environ.get("OLLAMA_BATCH_SIZE", "12"))
    max_candidates = int(os.environ.get("MAX_CANDIDATES", "72"))
    max_abs_chars = int(os.environ.get("MAX_ABS_TEXT_CHARS", "1200"))

    candidate_pool = [slim_candidate(c, max_abs_chars) for c in candidates[:max_candidates]]
    merged_entries = []
    for idx, batch in enumerate(chunks(candidate_pool, max(1, batch_size)), start=1):
        try:
            if backend == "openai":
                batch_entries = call_openai_batch(skill, batch, openai_model, timeout_sec, retries)
            else:
                batch_entries = call_ollama_batch(skill, batch, ollama_model, timeout_sec, retries)
            merged_entries.extend(batch_entries)
            print(f"[INFO] {backend} batch {idx}: {len(batch_entries)} entries")
        except Exception as err:
            print(f"[WARN] {backend} batch {idx} failed: {err}", file=sys.stderr)

    if not merged_entries:
        print(f"[WARN] {backend} produced no entries; continue with empty update.", file=sys.stderr)
        return []

    # Dedup by arXiv URL before downstream sanitation.
    out = []
    seen_urls = set()
    for e in merged_entries:
        if not isinstance(e, dict):
            continue
        u = str(e.get("arxiv_url", "")).strip()
        if not u or u in seen_urls:
            continue
        seen_urls.add(u)
        out.append(e)
    entries = out
    return entries if isinstance(entries, list) else []


def sanitize_entries(entries: list, candidate_map: dict, existing_ids: set):
    cleaned = []
    seen = set()
    for e in entries:
        if not isinstance(e, dict):
            continue
        title = " ".join(str(e.get("title", "")).split())
        short_name = " ".join(str(e.get("short_name", "")).split())
        venue = " ".join(str(e.get("venue", "")).split())
        arxiv_url = str(e.get("arxiv_url", "")).strip()
        code_url = str(e.get("code_url", "")).strip()

        m = re.match(r"^https://arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})(v\d+)?$", arxiv_url)
        if not (title and short_name and venue and m):
            continue
        arxiv_id = m.group(1)
        if arxiv_id in existing_ids or arxiv_id in seen:
            continue

        cand = candidate_map.get(arxiv_id, {})
        merged = " ".join([title, cand.get("title", ""), cand.get("summary", ""), cand.get("abs_text", "")])
        if not has_required_signals(merged):
            continue

        if not code_url:
            code_url = cand.get("code_url", "")
        if code_url and not re.match(r"^https?://", code_url):
            code_url = ""

        seen.add(arxiv_id)
        cleaned.append(
            {
                "short_name": short_name,
                "venue": venue,
                "title": title,
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
                "code_url": code_url,
            }
        )

    return cleaned[:20]


def build_block(entries: list) -> str:
    lines = [START, f"<!-- generated: {dt.date.today().isoformat()} -->", "", "### Auto MLLM Image Segmentation Updates", ""]
    if not entries:
        lines.append("No validated MLLM image segmentation entries in this cycle.")
    else:
        for i, e in enumerate(entries, 1):
            line = (
                f"{i}. **[{e['short_name']}]** | **{e['venue']}** | {e['title']} "
                f"| [`[pdf]`]({e['arxiv_url']})"
            )
            if e["code_url"]:
                line += f" | [`[code]`]({e['code_url']})"
            lines.append(line)
    lines.extend(["", END, ""])
    return "\n".join(lines)


def upsert_readme(readme: Path, block: str):
    text = readme.read_text(encoding="utf-8")
    s = text.find(START)
    e = text.find(END)
    if s != -1 and e != -1 and e > s:
        e2 = e + len(END)
        new_text = text[:s] + block.rstrip() + "\n" + text[e2:]
    else:
        add = (
            "\n## Auto MLLM Image Segmentation Updates\n\n"
            "This block is automatically updated every 3 days by a local agent.\n\n"
            + block
        )
        new_text = text.rstrip() + "\n" + add

    today = dt.date.today().isoformat()
    if "**Last Updated:" in new_text:
        new_text = re.sub(
            r"\*\*Last Updated:\s*[0-9]{4}-[0-9]{2}-[0-9]{2}\*\*",
            f"**Last Updated: {today}**",
            new_text,
            count=1,
        )
    readme.write_text(new_text, encoding="utf-8")


def run_git_push(repo: Path):
    if os.environ.get("AUTO_PUSH", "1") != "1":
        print("[INFO] AUTO_PUSH != 1, skip commit/push.")
        return

    def sh(cmd):
        return subprocess.run(cmd, cwd=repo, check=False, capture_output=True, text=True)

    diff = sh(["git", "diff", "--", "README.md"]) 
    if not diff.stdout.strip():
        print("[INFO] README unchanged. Skip commit/push.")
        return

    sh(["git", "add", "README.md"])
    commit = sh(["git", "commit", "-m", "chore: auto-update README (local ollama agent)"])
    if commit.returncode != 0 and "nothing to commit" not in (commit.stdout + commit.stderr).lower():
        print(commit.stdout)
        print(commit.stderr, file=sys.stderr)
        raise SystemExit(1)

    push = sh(["git", "push", "origin", "HEAD:main"])
    if push.returncode != 0:
        print(push.stdout)
        print(push.stderr, file=sys.stderr)
        raise SystemExit(1)
    print("[OK] README committed and pushed.")


def main():
    repo = Path(__file__).resolve().parents[1]
    readme = repo / "README.md"
    skill = repo / "SKILL.md"
    if not skill.exists():
        skill = repo / "skill.md"
    if not readme.exists():
        raise SystemExit("README.md not found.")

    all_items = []
    for kw in KEYWORDS:
        all_items.extend(query_arxiv(kw, 35))

    uniq = {}
    for p in all_items:
        uniq[p["arxiv_id"]] = p

    candidate_map = {}
    for arxiv_id, p in list(uniq.items())[:150]:
        try:
            meta = fetch_abs_metadata(arxiv_id)
        except Exception:
            meta = {"abs_url": p["arxiv_url"], "abs_text": "", "code_url": ""}
        candidate_map[arxiv_id] = {**p, **meta}

    existing_ids = set(re.findall(r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})", readme.read_text(encoding="utf-8"), flags=re.I))

    if skill.exists():
        skill_text = skill.read_text(encoding="utf-8")
    else:
        print("[WARN] SKILL.md not found. Use built-in curation rules.", file=sys.stderr)
        skill_text = DEFAULT_SKILL
    try:
        entries = call_llm(skill_text, list(candidate_map.values()))
    except Exception as err:
        print(f"[WARN] call_llm failed: {err}", file=sys.stderr)
        entries = []
    cleaned = sanitize_entries(entries, candidate_map, existing_ids)
    block = build_block(cleaned)
    upsert_readme(readme, block)
    run_git_push(repo)


if __name__ == "__main__":
    main()
