#!/usr/bin/env python3
"""Test how system prompt length affects format=json enforcement."""

import json
import requests

PROMPTS = {
    "267 chars": (
        'Return JSON: {"1":{"cv":N,"ar":N,"dd":N,"gu":N,"vc":N,"nq":N,"ss":N},...}\n'
        "Score 1-10: cv=config ar=architecture dd=data_def gu=usefulness vc=complexity nq=naming ss=scope\n"
        "LOW(1-3): loop vars, temp, generic results. HIGH(7-10): constants, framework, DB, loggers, types"
    ),
    "450 chars": (
        'Return JSON: {"1":{"cv":N,"ar":N,"dd":N,"gu":N,"vc":N,"nq":N,"ss":N},...}\n'
        "Score 1-10 per variable. Dimensions:\n"
        "cv=config/flag/URL/path/SQL ar=DB/router/logger/app/middleware dd=schema/type/enum/mapping\n"
        "gu=agent usefulness vc=literal=1,call=3,collection=7,complex=9 nq=1char=1,abbrev=3,desc=7,domain=9 ss=module=9,class=6,local=2,temp=1\n"
        "Most score LOW. DROP: loop counters, temp vars, result=func(). KEEP: constants, framework, DB, loggers, types."
    ),
    "600 chars": (
        'Return JSON: {"1":{"cv":N,"ar":N,"dd":N,"gu":N,"vc":N,"nq":N,"ss":N},...}\n'
        "Score variables for code search. Score 1-10 per dimension.\n"
        "cv: Config value, feature flag, URL, path, SQL, __all__\n"
        "ar: DB connection, router, logger, app instance, middleware\n"
        "dd: Schema, type alias, enum, named tuple, mapping, protocol\n"
        "gu: Would coding agent benefit from finding this?\n"
        "vc: Simple literal=1, function call=3, collection=7, complex=9\n"
        "nq: Single letter=1, abbreviation=3, descriptive=7, domain-specific=9\n"
        "ss: Module-level=9, class attr=6, function local=2, temp=1\n"
        "Most variables LOW. ~30% kept."
    ),
}

USER_MSG = "Score:\n1. [config.py] MAX_RETRIES = 3\n2. [app.py] x = 0"

for label, prompt in PROMPTS.items():
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3.5:4b",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": USER_MSG},
            ],
            "format": "json",
            "think": False,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 200},
        },
    )
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    stripped = content.strip()

    is_json = stripped.startswith("{")
    is_md = stripped.startswith("#") or stripped.startswith("```")
    valid = False
    try:
        json.loads(stripped.strip("`").lstrip("json\n"))
        valid = True
    except Exception:
        pass

    print(f"{label} ({len(prompt)} actual): JSON_start={is_json} MD_start={is_md} valid={valid}")
    print(f"  {repr(content[:150])}")
    print()
