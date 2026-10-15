#!/usr/bin/env python3
"""Greedy determinism probe against a running llama-server.

Sends the same prompts N times at temperature 0 and compares the returned
token ids byte for byte. Prints one JSON line per prompt and exits 1 on any
mismatch.

usage: determinism.py URL N [extra-json]
"""
import hashlib, json, os, sys, urllib.request

PROMPTS = [
    "<|im_start|>user\nImplement quicksort in Python with type hints.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nExplain why the sky is blue in three paragraphs.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWhat is 1234 * 5678? Show your work.<|im_end|>\n<|im_start|>assistant\n",
]


def run(url, prompt, extra):
    body = {"prompt": prompt, "n_predict": 256, "temperature": 0, "top_k": 1,
            "seed": 42, "cache_prompt": False, "return_tokens": True, **extra}
    req = urllib.request.Request(url + "/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    toks = r["tokens"]
    t = r.get("timings", {})
    return toks, t


def main():
    url, n = sys.argv[1], int(sys.argv[2])
    extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    bad = 0
    for i, p in enumerate(PROMPTS):
        hashes, tps, drafted, accepted = [], [], 0, 0
        for _ in range(n):
            toks, t = run(url, p, extra)
            hashes.append(hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16])
            tps.append(round(t.get("predicted_per_second", 0), 2))
            drafted += t.get("draft_n", 0)
            accepted += t.get("draft_n_accepted", 0)
        if os.environ.get("DUMP"):
            with open(f"{os.environ['DUMP']}.p{i}.json", "w") as f:
                json.dump(toks, f)
        same = len(set(hashes)) == 1
        bad += not same
        print(json.dumps({"prompt": i, "n": n, "identical": same, "hashes": sorted(set(hashes)),
                          "ntok": len(toks), "tok_s": tps, "draft_n": drafted,
                          "draft_accepted": accepted}), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
