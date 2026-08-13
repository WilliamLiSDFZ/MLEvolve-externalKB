#!/usr/bin/env bash
# Does the CLIProxyAPI instance hold up under MLEvolve's call pattern, or does it hit a
# rate limit partway through a run?
#
# The smoke test proved the proxy answers 25 one-off requests. That says nothing about a
# 12-hour run, which issues thousands of calls with large prompts against a SUBSCRIPTION
# quota (the proxy log shows `provider=codex`, i.e. the ChatGPT plan, not metered API keys).
# The failure this test exists to find is the expensive one: limits that engage after a while
# and turn a 12 GPU-hour run into a stream of retries.
#
# Two phases, reported separately:
#   BURST  short window at maximum concurrency -> finds the instantaneous ceiling
#   PACED  the rest of the window at MLEvolve's real rate -> finds sustained-use limits
#
# Env:
#   CLIPROXY_API_KEY   required
#   PROXY_BASE_URL     talk to an existing proxy instead of starting one (e.g. a Service)
#   DURATION_SECS      total test length            (default 900 = 15 min)
#   BURST_SECS         length of the burst phase    (default 60)
#   WORKERS            concurrency                  (default 3 = agent.search.parallel_search_num)
#   PACED_CALLS_PER_MIN target rate in phase 2      (default 12, ~ what a real run sustains)
#   PROMPT_TOKENS      approximate input size       (default 3000; real prompts run larger)
#   TEST_MODEL         model to hammer              (default gpt-5.6-terra)
set -u

PROXY_DIR=${PROXY_DIR:-/workspace/cliproxy}
PORT=${PROXY_PORT:-8317}
: "${CLIPROXY_API_KEY:?CLIPROXY_API_KEY is not set — create the cliproxy-key Secret}"

if [ -z "${PROXY_BASE_URL:-}" ]; then
    cd "$PROXY_DIR" || { echo "FAIL: $PROXY_DIR not found on the PVC"; exit 1; }
    [ -x ./cli-proxy-api ] || chmod +x ./cli-proxy-api
    echo "=== starting a local proxy (set PROXY_BASE_URL to test a shared one instead) ==="
    ./cli-proxy-api --config config.yaml > /tmp/proxy.log 2>&1 &
    PROXY_PID=$!
    trap 'kill "$PROXY_PID" 2>/dev/null || true' EXIT
    for i in $(seq 1 60); do
        python3 - "$PORT" <<'PY' 2>/dev/null && break
import socket, sys
s = socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
        sleep 1
    done
    kill -0 "$PROXY_PID" 2>/dev/null || { echo "FAIL: proxy exited"; cat /tmp/proxy.log; exit 1; }
    export PROXY_BASE_URL="http://127.0.0.1:${PORT}/v1"
fi
echo "target: $PROXY_BASE_URL"

python3 - <<'PY'
import collections, json, os, random, statistics, sys, threading, time
import urllib.error, urllib.request

BASE     = os.environ["PROXY_BASE_URL"].rstrip("/")
KEY      = os.environ["CLIPROXY_API_KEY"]
MODEL    = os.environ.get("TEST_MODEL", "gpt-5.6-terra")
DURATION = int(os.environ.get("DURATION_SECS", 900))
BURST    = int(os.environ.get("BURST_SECS", 60))
WORKERS  = int(os.environ.get("WORKERS", 3))
RATE     = float(os.environ.get("PACED_CALLS_PER_MIN", 12))
PROMPT_TOKENS = int(os.environ.get("PROMPT_TOKENS", 3000))

HDRS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
IS_REASONING = any(MODEL.lower().startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))
TOK_KEY = "max_completion_tokens" if IS_REASONING else "max_tokens"

# A prompt shaped like the ones MLEvolve sends: a chunk of code plus an instruction. Token
# volume is what a subscription meters, so testing with toy prompts would understate load by
# an order of magnitude.
_FILLER = ("    df['feat_%d'] = df['text'].str.len() * %d  # engineered feature\n"
           % (0, 0)).join("" for _ in range(0))
CODE = "\n".join(f"    df['feat_{i}'] = df['text'].str.len() * {i}" for i in range(PROMPT_TOKENS // 12))
PROMPT = ("Below is a solution for a multi-label text classification competition.\n"
          "```python\ndef build_features(df):\n" + CODE + "\n    return df\n```\n"
          "Reply with exactly one word: OK")

records = []          # (t_start, phase, latency, status)  status: "ok" | "429" | "5xx" | ...
rec_lock = threading.Lock()
stop_at = time.time() + DURATION
phase_switch = time.time() + BURST


def one_call():
    payload = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}], TOK_KEY: 8}
    if IS_REASONING:
        payload["reasoning_effort"] = "high"
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(payload).encode(), headers=HDRS, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            body = json.loads(r.read())
        return time.time() - t0, "ok", (body.get("usage") or {}).get("total_tokens", 0)
    except urllib.error.HTTPError as e:
        code = e.code
        kind = "429" if code == 429 else ("5xx" if code >= 500 else f"{code}")
        try:
            msg = e.read()[:160].decode(errors="replace")
        except Exception:
            msg = ""
        return time.time() - t0, kind, 0 if not msg else 0
    except Exception as e:
        return time.time() - t0, type(e).__name__, 0


def worker():
    # In the paced phase each worker sleeps so the fleet approximates RATE calls/min.
    gap = (60.0 * WORKERS / RATE) if RATE > 0 else 0
    while time.time() < stop_at:
        t0 = time.time()
        phase = "burst" if t0 < phase_switch else "paced"
        lat, status, toks = one_call()
        with rec_lock:
            records.append((t0, phase, lat, status, toks))
        if phase == "paced":
            time.sleep(max(0.0, gap - lat + random.uniform(-1, 1)))


print(f"model={MODEL}  workers={WORKERS}  duration={DURATION}s "
      f"(burst {BURST}s, then paced at ~{RATE}/min)")
print(f"prompt ~{len(PROMPT)//4} tokens\n")
print(f"{'elapsed':>8}{'calls':>7}{'ok':>6}{'429':>6}{'other':>7}{'p50 s':>8}{'tok/min':>9}")
print("-" * 51)

threads = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
for t in threads:
    t.start()

start = time.time()
last = 0
while any(t.is_alive() for t in threads):
    time.sleep(30)
    with rec_lock:
        snap = list(records)
    window = [r for r in snap[last:]]
    last = len(snap)
    if not window:
        continue
    lats = [r[2] for r in window if r[3] == "ok"]
    n429 = sum(1 for r in window if r[3] == "429")
    other = sum(1 for r in window if r[3] not in ("ok", "429"))
    toks = sum(r[4] for r in window)
    print(f"{time.time()-start:7.0f}s{len(window):>7}{len(lats):>6}{n429:>6}{other:>7}"
          f"{(statistics.median(lats) if lats else 0):>8.2f}{toks*2:>9}")

print("\n" + "=" * 62)
with rec_lock:
    snap = list(records)
if not snap:
    print("FAIL: no calls completed at all")
    sys.exit(1)

for phase in ("burst", "paced"):
    rows = [r for r in snap if r[1] == phase]
    if not rows:
        continue
    ok = [r[2] for r in rows if r[3] == "ok"]
    span = max(r[0] for r in rows) - min(r[0] for r in rows) + 1
    by_status = collections.Counter(r[3] for r in rows)
    print(f"\n{phase.upper()}  {len(rows)} calls over {span:.0f}s "
          f"= {len(rows)*60/span:.1f} calls/min")
    print(f"  statuses: {dict(by_status)}")
    if ok:
        ok_sorted = sorted(ok)
        p = lambda q: ok_sorted[min(len(ok_sorted) - 1, int(len(ok_sorted) * q))]
        print(f"  latency: p50 {p(.5):.2f}s  p95 {p(.95):.2f}s  max {ok_sorted[-1]:.2f}s")
    print(f"  tokens: {sum(r[4] for r in rows):,}")

# Did failures cluster late? That is the signature of a quota engaging, as opposed to
# sporadic upstream flakiness.
minutes = collections.defaultdict(lambda: [0, 0])
t0 = min(r[0] for r in snap)
for r in snap:
    m = minutes[int((r[0] - t0) // 60)]
    m[0] += 1
    if r[3] != "ok":
        m[1] += 1
bad = [(m, v) for m, v in sorted(minutes.items()) if v[1]]
print(f"\nerrors by minute: {'none' if not bad else ''}")
for m, (tot, err) in bad:
    print(f"  min {m:>3}: {err}/{tot} failed")

total = len(snap)
errs = sum(1 for r in snap if r[3] != "ok")
rate_429 = sum(1 for r in snap if r[3] == "429")
print(f"\nTOTAL {total} calls, {errs} failed ({errs/total*100:.1f}%), "
      f"{sum(r[4] for r in snap):,} tokens")

if rate_429:
    first = min(r[0] for r in snap if r[3] == "429") - t0
    print(f"\nVERDICT: RATE LIMITED — first 429 at {first:.0f}s in. "
          f"A 12 h run will stall. Lower agent.search.parallel_search_num, or keep the "
          f"metered API for the agent and use the proxy only for cheap calls.")
    sys.exit(1)
if errs / total > 0.05:
    print(f"\nVERDICT: UNSTABLE — {errs/total*100:.0f}% failures without 429s. "
          f"Check the statuses above before trusting this for a long run.")
    sys.exit(1)
print(f"\nVERDICT: no rate limiting seen at {WORKERS}-way concurrency for {DURATION//60} min.")
print("  Caveat: this cannot rule out a quota that engages after hours. Before committing a")
print("  12 h run, do a 1 h MLEvolve shakedown (TIME_LIMIT_SECS=3600) through the proxy.")
PY
STATUS=$?
[ -f /tmp/proxy.log ] && { echo "=== proxy log (tail) ==="; tail -15 /tmp/proxy.log; }
exit $STATUS
