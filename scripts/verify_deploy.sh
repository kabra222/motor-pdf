#!/usr/bin/env bash
set -euo pipefail

URL="${1:-}"
if [ -z "$URL" ]; then
  URL="https://motor-pdf-production.up.railway.app"
  echo "ℹ️  Usando URL padrão: $URL"
fi

PASS=0
FAIL=0

check() {
  local label="$1"
  shift
  local out
  out=$("$@" 2>&1) || {
    echo "  ❌ $label"
    echo "     $out"
    FAIL=$((FAIL + 1))
    return
  }
  echo "  ✅ $label"
  PASS=$((PASS + 1))
}

check_pipeline() {
  local label="$1"
  shift
  local out
  out=$(eval "$@" 2>&1) || {
    echo "  ❌ $label"
    echo "     $out"
    FAIL=$((FAIL + 1))
    return
  }
  echo "  ✅ $label"
  PASS=$((PASS + 1))
}

echo ""
echo "══════════════════════════════════════════"
echo "  Verify Deploy — $URL"
echo "══════════════════════════════════════════"
echo ""

# 1. Health endpoint
check_pipeline "health endpoint" \
  "curl -sf '$URL/api/v1/health' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'; assert d['version']\""

# 2. JavaScript syntax (fetch homepage, extract <script>, compile with node)
check_pipeline "JavaScript syntax" \
  "curl -sf '$URL/' | node -e \"
const fs = require('fs');
const html = fs.readFileSync('/dev/stdin', 'utf-8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) { console.error('no script tag'); process.exit(1); }
try { new Function(m[1]); } catch(e) { console.error(e.message); process.exit(1); }
\""

# 3. Extract with a generated PDF
check "extract PDF" \
  uv run python3 -c "
import tempfile, fitz, sys, os, urllib.request
tmp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
doc = fitz.open()
doc.new_page().insert_text((50, 750), 'Test verify deploy', fontsize=12)
doc.save(tmp.name); doc.close()
import requests
with open(tmp.name, 'rb') as f:
    r = requests.post('$URL/api/v1/extract', files={'file': f})
assert r.status_code == 200, f'status {r.status_code}'
d = r.json()
assert d['num_pages'] == 1, f'{d[\"num_pages\"]} pages'
assert len(d['text']) > 0, 'empty text'
os.unlink(tmp.name)
"

# 4. Agent status
check_pipeline "agent status" \
  "curl -sf '$URL/api/v1/agent/status' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert 'provider' in d\""

# 5. Cache stats
check_pipeline "cache stats" \
  "curl -sf '$URL/api/v1/cache/stats' | python3 -c \"import sys,json; json.load(sys.stdin)\""

# 6. Swagger docs
check_pipeline "swagger docs (200)" \
  "curl -sf -o /dev/null -w '%{http_code}' '$URL/docs' | grep -q 200"

# 7. Web UI serves index.html with expected title
check_pipeline "web UI title" \
  "curl -sf '$URL/' | python3 -c \"import sys; assert '<title>Motor PDF' in sys.stdin.read()\""

echo ""
echo "══════════════════════════════════════════"
echo "  Resultados: $PASS passed, $FAIL failed"
echo "══════════════════════════════════════════"
exit $FAIL
