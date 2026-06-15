"""E1: Collect LLM responses to the Sleep-Persona Probe.

5 models x 152 prompts x N replicates, async with on-disk caching so the run is
resumable and reproducible. Each (model, prompt_id, replicate) is one record
keyed deterministically; already-collected keys are skipped.

Output: results/model_outputs/raw_responses.jsonl  (append-only, one JSON/line)
"""
import asyncio, json, os, sys, time
import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

OUT_PATH = os.path.join(C.MODEL_OUT_DIR, "raw_responses.jsonl")


def load_prompts():
    with open(C.PROMPTS_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("content") and not r.get("error"):
                        done.add((r["model_label"], r["prompt_id"], r["replicate"]))
                except Exception:
                    continue
    return done


async def call_one(client, sem, model_label, model_id, item, rep, out_f, lock):
    key = (model_label, item["id"], rep)
    seed = C.SEED_BASE + rep
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": C.GEN_TEMPERATURE,
        "max_tokens": C.GEN_MAX_TOKENS,
        "seed": seed,
    }
    async with sem:
        last_err = None
        for attempt in range(C.MAX_RETRIES):
            try:
                resp = await client.post(
                    f"{C.OPENROUTER_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {C.OPENROUTER_KEY}"},
                    json=payload, timeout=C.REQUEST_TIMEOUT,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_err = f"HTTP {resp.status_code}"
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                rec = {
                    "model_label": model_label, "model_id": model_id,
                    "prompt_id": item["id"], "scenario": item["scenario"],
                    "persona_id": item["persona_id"], "factor": item["factor"],
                    "level": item["level"], "is_control": item["is_control"],
                    "replicate": rep, "seed": seed,
                    "content": content,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "error": None, "ts": time.time(),
                }
                async with lock:
                    out_f.write(json.dumps(rec) + "\n"); out_f.flush()
                return True
            except Exception as e:
                last_err = str(e)[:200]
                await asyncio.sleep(min(2 ** attempt, 30))
        # give up: record the failure
        rec = {"model_label": model_label, "model_id": model_id, "prompt_id": item["id"],
               "scenario": item["scenario"], "persona_id": item["persona_id"],
               "factor": item["factor"], "level": item["level"],
               "is_control": item["is_control"], "replicate": rep, "seed": seed,
               "content": None, "error": last_err, "ts": time.time()}
        async with lock:
            out_f.write(json.dumps(rec) + "\n"); out_f.flush()
        print(f"  FAIL {key} -> {last_err}", flush=True)
        return False


async def main():
    assert C.OPENROUTER_KEY, "OPENROUTER_KEY not set"
    prompts = load_prompts()
    done = load_done(OUT_PATH)
    print(f"Loaded {len(prompts)} prompts; {len(done)} (model,id,rep) already done.")

    tasks_spec = []
    for label, mid in C.MODELS.items():
        for item in prompts:
            for rep in range(C.N_REPLICATES):
                if (label, item["id"], rep) not in done:
                    tasks_spec.append((label, mid, item, rep))
    total = len(C.MODELS) * len(prompts) * C.N_REPLICATES
    print(f"To collect: {len(tasks_spec)} of {total} total.")

    sem = asyncio.Semaphore(C.MAX_CONCURRENCY)
    lock = asyncio.Lock()
    t0 = time.time()
    with open(OUT_PATH, "a") as out_f:
        async with httpx.AsyncClient() as client:
            coros = [call_one(client, sem, lbl, mid, item, rep, out_f, lock)
                     for (lbl, mid, item, rep) in tasks_spec]
            ok = 0
            for i, fut in enumerate(asyncio.as_completed(coros), 1):
                ok += await fut
                if i % 100 == 0:
                    print(f"  {i}/{len(tasks_spec)} done ({ok} ok), {time.time()-t0:.0f}s", flush=True)
    print(f"Finished in {time.time()-t0:.0f}s. Output -> {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
