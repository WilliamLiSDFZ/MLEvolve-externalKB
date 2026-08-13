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

print(f"{len(models)} models exposed:")
for m in models[:25]:
    print(f"    {m}")
if len(models) > 25:
    print(f"    ... and {len(models) - 25} more")
if not models:
    print("FAIL: proxy is up but exposes no models — check auth/ and config.yaml")
    sys.exit(1)

# Prefer whatever the agent would actually use, else fall back to the first model offered.
want = os.environ.get("TEST_MODEL", "")
target = want if want in models else next(
    (m for m in models if any(k in m.lower() for k in ("sonnet", "opus", "gpt-5", "codex"))),
    models[0])
print(f"\ncalling chat/completions with model={target!r} ...")

try:
    r = call("/chat/completions", {
        "model": target,
        "messages": [{"role": "user",
                      "content": "Reply with exactly the word: PROXY_OK"}],
        "max_tokens": 16,
    })
except urllib.error.HTTPError as e:
    body = e.read()[:600].decode(errors="replace")
    print(f"FAIL: HTTP {e.code}: {body}")
    if e.code == 400 and "max_tokens" in body:
        print("  -> this model wants max_completion_tokens; MLEvolve handles that in "
              "llm/model_profiles.py, but this smoke test does not.")
    sys.exit(1)
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    sys.exit(1)

text = (r.get("choices") or [{}])[0].get("message", {}).get("content", "")
usage = r.get("usage", {})
print(f"reply : {text!r}")
print(f"usage : {usage}")
print(f"\nPASS — proxy reachable, authenticated, and completing with {target!r}")
print("\nTo point MLEvolve at it, set in the mlevolve-llm Secret:")
print(f"    LLM_BASE_URL=http://<service>:{os.environ.get('PROXY_PORT','8317')}/v1")
print(f"    LLM_MODEL={target}")
print( "    LLM_API_KEY=<one of config.yaml's api-keys>")
PY
STATUS=$?

echo "=== proxy log (tail) ==="
tail -25 /tmp/proxy.log
exit $STATUS
