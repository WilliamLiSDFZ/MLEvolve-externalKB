#!/usr/bin/env bash
# Smoke-test the CLIProxyAPI instance installed on the PVC at /workspace/cliproxy.
#
# Starts the proxy inside THIS pod (the binary and the auth files live on the shared PVC),
# waits for it to listen, then makes two real calls against it: GET /v1/models and a minimal
# POST /v1/chat/completions. Exits non-zero if either fails, so the Job's status is the result.
#
# Why start it in-pod rather than talking to the dev pod: a Job pod has its own network
# namespace, so `localhost:8317` on the dev pod is unreachable. Reaching the dev pod would
# need a Service (see the note at the bottom of k8s/job-cliproxy-test.yaml). For a smoke test,
# self-contained is simpler and also proves the binary runs outside the dev pod.
#
# Requires: CLIPROXY_API_KEY in the environment (from the cliproxy-key Secret).
set -u

PROXY_DIR=${PROXY_DIR:-/workspace/cliproxy}
PORT=${PROXY_PORT:-8317}

: "${CLIPROXY_API_KEY:?CLIPROXY_API_KEY is not set — create the cliproxy-key Secret}"

cd "$PROXY_DIR" || { echo "FAIL: $PROXY_DIR not found on the PVC"; exit 1; }
[ -x ./cli-proxy-api ] || chmod +x ./cli-proxy-api

echo "=== auth entries on the PVC ==="
ls -1 auth/ 2>/dev/null || echo "  (no auth/ directory — the proxy will have no upstream accounts)"

echo "=== starting proxy ==="
./cli-proxy-api --config config.yaml > /tmp/proxy.log 2>&1 &
PROXY_PID=$!
trap 'kill "$PROXY_PID" 2>/dev/null || true' EXIT

for i in $(seq 1 60); do
    if python3 - "$PORT" <<'PY' 2>/dev/null
import socket, sys
s = socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
    then
        echo "proxy listening on :$PORT after ${i}s"
        break
    fi
    sleep 1
done

if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "FAIL: proxy exited during startup. Log:"; cat /tmp/proxy.log; exit 1
fi

echo "=== calling the proxy ==="
python3 - <<'PY'
import json, os, sys, urllib.error, urllib.request

BASE = f"http://127.0.0.1:{os.environ.get('PROXY_PORT', '8317')}/v1"
KEY  = os.environ["CLIPROXY_API_KEY"]
HDRS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def call(path, payload=None, timeout=180):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=HDRS,
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


try:
    models = [m["id"] for m in call("/models").get("data", [])]
except urllib.error.HTTPError as e:
    print(f"FAIL: GET /v1/models -> HTTP {e.code}: {e.read()[:400].decode(errors='replace')}")
    print("  401/403 means the key does not match config.yaml's api-keys list.")
    sys.exit(1)
except Exception as e:
    print(f"FAIL: GET /v1/models -> {type(e).__name__}: {e}")
    sys.exit(1)

print(f"{len(models)} models advertised by the proxy.")
if not models:
    print("FAIL: proxy is up but exposes no models — check auth/ and config.yaml")
    sys.exit(1)

# The advertised list comes from a static models.json the proxy refreshes from GitHub, NOT
# from what the linked account can actually call — so a name appearing here means nothing.
# Probe each one. The inventory is the point of this test: it is what decides LLM_MODEL.
SKIP = ("gpt-image", "codex-auto-review", "embedding", "whisper", "tts")
candidates = [m for m in models if not any(s in m.lower() for s in SKIP)]
want = os.environ.get("TEST_MODEL", "")
if want:                                   # test the named model first if one was given
    candidates = [want] + [m for m in candidates if m != want]

PROMPT = [{"role": "user", "content": "Reply with exactly the word: PROXY_OK"}]


def probe(model):
    """Return (ok, note). Retries with max_completion_tokens if max_tokens is rejected."""
    for tok_key in ("max_tokens", "max_completion_tokens"):
        try:
            r = call("/chat/completions",
                     {"model": model, "messages": PROMPT, tok_key: 16}, timeout=120)
            txt = (r.get("choices") or [{}])[0].get("message", {}).get("content", "")
            u = r.get("usage", {}) or {}
            note = f"{tok_key}, {u.get('total_tokens', '?')} tok, reply {txt.strip()[:24]!r}"
            return True, note
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode(errors="replace")
            if e.code == 400 and "max_tokens" in body and tok_key == "max_tokens":
                continue                    # this model wants max_completion_tokens
            short = body.split('"message":')[-1][:110].strip(' "}')
            return False, f"HTTP {e.code}: {short}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
    return False, "rejected both max_tokens and max_completion_tokens"


working, broken = [], []
print(f"\nprobing {len(candidates)} chat models (image/review models skipped):\n")
for m in candidates:
    ok, note = probe(m)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {m:<32} {note}")
    (working if ok else broken).append((m, note))

print(f"\n{len(working)} of {len(candidates)} models usable by this account.")
if not working:
    print("FAIL: proxy is reachable and authenticated, but no advertised model can be "
          "called. The account behind auth/ has access to none of them.")
    sys.exit(1)

names = [m for m, _ in working]
needs_mct = [m for m, n in working if "max_completion_tokens" in n]
print("\nUSABLE MODELS:")
for m, n in working:
    print(f"    {m}")
if needs_mct:
    print("\nThese require max_completion_tokens rather than max_tokens:")
    for m in needs_mct:
        print(f"    {m}")
    print("  -> llm/model_profiles.py chooses that by NAME PREFIX; check "
          "_MAX_COMPLETION_TOKENS_PREFIXES covers them before switching MLEvolve over.")

# Prefer the model MLEvolve already runs on: if it works through the proxy, switching over
# is a Secret change and nothing else. Otherwise fall back to the same family, then anything.
CURRENT = os.environ.get("LLM_MODEL_HINT", "gpt-5.6-terra")
pick = (CURRENT if CURRENT in names
        else next((m for m in names if m.startswith("gpt-5.6")),
                  next((m for m in names if m.startswith("gpt-5")), names[0])))
if pick == CURRENT:
    print(f"\n{CURRENT} works through the proxy — this is a drop-in swap, "
          "no model_profiles.py change needed.")
else:
    print(f"\nNote: {CURRENT} (MLEvolve's current model) is NOT usable here; "
          f"nearest working alternative is {pick}.")
print(f"\nPASS — proxy reachable, authenticated, {len(working)} models working.")
print("\nTo point MLEvolve at it, set in the mlevolve-llm Secret:")
print(f"    LLM_BASE_URL=http://<service>:{os.environ.get('PROXY_PORT','8317')}/v1")
print(f"    LLM_MODEL={pick}")
print( "    LLM_API_KEY=<one of config.yaml's api-keys>")
PY
STATUS=$?

echo "=== proxy log (tail) ==="
tail -25 /tmp/proxy.log
exit $STATUS
