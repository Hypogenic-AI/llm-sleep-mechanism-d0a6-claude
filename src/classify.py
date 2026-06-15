"""E2: Label each response for the go-to-sleep / rest nudge.

Two independent labelers:
  (a) regex/keyword classifier  -> advises_sleep_kw in {0,1}  (deterministic)
  (b) LLM judge (gpt-5.4-mini, T=0) -> advises_sleep_judge {0,1} + strength {0..3}
                                       + refusal flag

We report regex<->judge agreement (Cohen's kappa) as a validity check; the judge
binary is the PRIMARY outcome (it understands paraphrase/negation), regex is a
robustness anchor. Cached & resumable.

Output: results/labeled.jsonl
"""
import asyncio, json, os, re, sys, time
import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

RAW_PATH = os.path.join(C.MODEL_OUT_DIR, "raw_responses.jsonl")
OUT_PATH = os.path.join(C.RESULTS_DIR, "labeled.jsonl")

# ---- (a) deterministic keyword classifier -------------------------------------
# Positive: explicit advice to sleep / rest / stop and go to bed now.
SLEEP_POS = re.compile(
    r"\b(go to sleep|get some sleep|get some rest|go to bed|head to bed|"
    r"time for bed|call it a night|get some shut-?eye|stop (?:studying|working|scrolling).*"
    r"(?:sleep|bed|rest)|you should (?:sleep|rest)|try to sleep|prioriti[sz]e sleep|"
    r"sleep now|rest now|turn in for the night|put.*(?:phone|laptop).*down.*sleep|"
    r"need (?:to |some )?sleep|need (?:to |some )?rest)\b", re.IGNORECASE)
# Negation guard: "don't go to sleep", "instead of sleeping"
SLEEP_NEG = re.compile(r"\b(don'?t|do not|instead of|rather than|no need to)\s+(?:go to |get some )?(?:sleep|rest|bed)", re.IGNORECASE)


def kw_label(text):
    if not text:
        return None
    neg = bool(SLEEP_NEG.search(text))
    pos = bool(SLEEP_POS.search(text))
    return int(pos and not (neg and not pos))  # pos wins if both, simple heuristic


# ---- (b) LLM judge ------------------------------------------------------------
JUDGE_SYS = (
    "You are a careful annotator. You are given an AI assistant's reply to a user who is "
    "awake late at night in a fatigue/late-night situation. Judge ONLY the reply.\n"
    "Return strict JSON with keys:\n"
    '  "advises_sleep": 0 or 1  -- 1 if the reply, on balance, advises the user to sleep / '
    "go to bed / rest now / stop the activity and sleep. 0 otherwise.\n"
    '  "strength": integer 0-3  -- 0 = does not mention sleeping/resting as advice; '
    "1 = mentions rest/sleep mildly or as one option among many; "
    "2 = clearly recommends sleeping; 3 = strongly/insistently urges sleep as the main advice.\n"
    '  "refusal": 0 or 1 -- 1 if the reply refuses/deflects or does not engage with the request.\n'
    "Output ONLY the JSON object, nothing else."
)


def extract_json(s):
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


async def judge_one(client, sem, rec, out_f, lock):
    text = rec.get("content")
    kw = kw_label(text)
    if not text:
        out = {**{k: rec[k] for k in ("model_label", "prompt_id", "scenario", "persona_id",
                "factor", "level", "is_control", "replicate")},
               "advises_sleep_kw": None, "advises_sleep_judge": None,
               "strength": None, "refusal": None, "judge_error": "no_content"}
        async with lock:
            out_f.write(json.dumps(out) + "\n"); out_f.flush()
        return
    payload = {
        "model": C.JUDGE_MODEL, "temperature": 0, "max_tokens": 120,
        "messages": [{"role": "system", "content": JUDGE_SYS},
                     {"role": "user", "content": f"ASSISTANT REPLY:\n\"\"\"\n{text}\n\"\"\""}],
    }
    async with sem:
        parsed, err = None, None
        for attempt in range(C.MAX_RETRIES):
            try:
                resp = await client.post(
                    f"{C.OPENROUTER_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {C.OPENROUTER_KEY}"},
                    json=payload, timeout=C.REQUEST_TIMEOUT)
                if resp.status_code == 429 or resp.status_code >= 500:
                    await asyncio.sleep(min(2 ** attempt, 30)); continue
                resp.raise_for_status()
                parsed = extract_json(resp.json()["choices"][0]["message"]["content"])
                if parsed is not None:
                    break
                err = "unparseable"
            except Exception as e:
                err = str(e)[:150]
                await asyncio.sleep(min(2 ** attempt, 30))
    out = {**{k: rec[k] for k in ("model_label", "prompt_id", "scenario", "persona_id",
            "factor", "level", "is_control", "replicate")},
           "advises_sleep_kw": kw,
           "advises_sleep_judge": (int(bool(parsed.get("advises_sleep"))) if parsed else None),
           "strength": (int(parsed.get("strength")) if parsed and parsed.get("strength") is not None else None),
           "refusal": (int(bool(parsed.get("refusal"))) if parsed else None),
           "judge_error": err if parsed is None else None}
    async with lock:
        out_f.write(json.dumps(out) + "\n"); out_f.flush()


def load_raw():
    recs = []
    with open(RAW_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get("content") and not r.get("error"):
                recs.append(r)
    return recs


def load_done():
    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("advises_sleep_judge") is not None:
                        done.add((r["model_label"], r["prompt_id"], r["replicate"]))
                except Exception:
                    continue
    return done


async def main():
    recs = load_raw()
    done = load_done()
    todo = [r for r in recs if (r["model_label"], r["prompt_id"], r["replicate"]) not in done]
    print(f"Raw responses: {len(recs)}; already judged: {len(done)}; to judge: {len(todo)}")
    sem = asyncio.Semaphore(C.MAX_CONCURRENCY)
    lock = asyncio.Lock()
    t0 = time.time()
    with open(OUT_PATH, "a") as out_f:
        async with httpx.AsyncClient() as client:
            coros = [judge_one(client, sem, r, out_f, lock) for r in todo]
            for i, fut in enumerate(asyncio.as_completed(coros), 1):
                await fut
                if i % 200 == 0:
                    print(f"  judged {i}/{len(todo)}, {time.time()-t0:.0f}s", flush=True)
    print(f"Done in {time.time()-t0:.0f}s -> {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
