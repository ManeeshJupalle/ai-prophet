"""Lightweight dashboard for the Prophet Arena trade benchmark.

Serves HTML at / and proxies /api/* to the core API so CORS and the
API key are handled in-process (the key never reaches the browser).
The slug filter is baked into the HTML at generation time.

Standalone usage (requires only ai-prophet-core)::

    prophet-dashboard --slug my_experiment

    # or equivalently
    python -m ai_prophet_core.dashboard --slug my_experiment

Via the CLI package (re-exported as ``prophet trade dashboard``)::

    prophet trade dashboard --slug my_experiment
    prophet trade eval run -m openai:gpt-5.2 --slug test --dashboard

The dashboard reads ``PA_SERVER_URL`` and ``PA_SERVER_API_KEY`` from the
environment when CLI args are not supplied.
"""

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import httpx

from .client import DEFAULT_API_URL

DEFAULT_REPORTING_API_URL = "https://trade-ui-api-998105805337.us-central1.run.app"

_API_URL = ""
_API_KEY = ""
_REPORTING_API_URL = ""
_SLUG = ""
_HTML_BYTES = b""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._proxy_core(parsed.path[4:], parsed.query)
        elif parsed.path.startswith("/report/"):
            self._proxy_reporting(parsed.path[7:], parsed.query)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_HTML_BYTES)

    def _proxy_core(self, path, query):
        self._proxy(base_url=_API_URL, path=path, query=query, api_key=_API_KEY, scope_slug=True)

    def _proxy_reporting(self, path, query):
        self._proxy(base_url=_REPORTING_API_URL, path=path, query=query, api_key="", scope_slug=False)

    def _proxy(self, *, base_url, path, query, api_key, scope_slug):
        url = f"{base_url}{path}"
        if query:
            url += f"?{query}"
        try:
            headers = {"X-API-Key": api_key} if api_key else None
            resp = httpx.get(url, timeout=15, headers=headers)
            data = resp.content
            # Scope /experiments to configured slug
            if scope_slug and _SLUG and path == "/experiments" and resp.status_code == 200:
                items = resp.json()
                if isinstance(items, list):
                    data = json.dumps([e for e in items if e.get("experiment_slug") == _SLUG]).encode()
                elif isinstance(items, dict) and isinstance(items.get("experiments"), list):
                    scoped = {
                        **items,
                        "experiments": [
                            e for e in items["experiments"] if e.get("experiment_slug") == _SLUG
                        ],
                    }
                    data = json.dumps(scoped).encode()
            self.send_response(resp.status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def log_message(self, fmt, *args):
        pass


def open_dashboard(
    api_url: str,
    slug: str = "",
    port: int = 8501,
    api_key: str | None = None,
    reporting_url: str | None = None,
    reporting_api_url: str | None = None,
    *,
    block: bool = False,
):
    """Serve the dashboard and open it in the browser.

    ``block=True`` is used by the standalone dashboard command so the local
    HTTP server stays alive until the user stops it. ``block=False`` keeps the
    dashboard as a sidecar during ``prophet trade eval run --dashboard``.
    """
    global _API_URL, _API_KEY, _REPORTING_API_URL, _SLUG, _HTML_BYTES
    _API_URL = api_url.rstrip("/")
    _API_KEY = api_key or ""
    _REPORTING_API_URL = (
        reporting_url
        or reporting_api_url
        or os.environ.get("PA_TRADE_UI_API_URL")
        or os.environ.get("PA_REPORTING_API_URL")
        or DEFAULT_REPORTING_API_URL
    ).rstrip("/")
    _SLUG = slug
    _HTML_BYTES = _HTML.replace("__REQUESTED_SLUG__", json.dumps(slug)).encode()

    server = HTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    webbrowser.open(f"http://localhost:{port}")
    print(f"  Dashboard: http://localhost:{port}")
    print(f"  Core API:  {_API_URL}")
    print(f"  Report API: {_REPORTING_API_URL}")
    if slug:
        print(f"  Experiment: {slug}")

    if block:
        try:
            print("  Press Ctrl+C to stop the dashboard")
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard stopped")
        finally:
            server.server_close()


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point: ``prophet-dashboard`` and ``python -m ai_prophet_core.dashboard``."""
    parser = argparse.ArgumentParser(
        prog="prophet-dashboard",
        description="Live dashboard for the Prophet Arena trade benchmark.",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("PA_SERVER_URL") or DEFAULT_API_URL,
        help="Core API base URL (env: PA_SERVER_URL).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PA_SERVER_API_KEY"),
        help="Prophet Arena API key (env: PA_SERVER_API_KEY).",
    )
    parser.add_argument(
        "--slug",
        default="",
        help="Filter the dashboard to a single experiment slug.",
    )
    parser.add_argument(
        "--reporting-url",
        default=os.environ.get("PA_REPORTING_API_URL"),
        help=(
            "Read-only reporting API for PnL history and leaderboard "
            "(env: PA_REPORTING_API_URL). Defaults to the hosted "
            "trade-ui-api service."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Local port to bind (default: 8501).",
    )
    args = parser.parse_args(argv)

    if not args.api_key:
        parser.error("missing API key: pass --api-key or set PA_SERVER_API_KEY")

    open_dashboard(
        api_url=args.api_url,
        slug=args.slug,
        port=args.port,
        api_key=args.api_key,
        reporting_url=args.reporting_url,
        block=True,
    )
    return 0


# ---------------------------------------------------------------------------
# Self-contained HTML -- uses the local /api proxy so CORS/API keys stay hidden
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trade Benchmark Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#09090b;--panel:#111113;--panel2:#18181b;--fg:#f4f4f5;--muted:#71717a;
  --sub:#a1a1aa;--border:#27272a;--soft:#3f3f46;--green:#22c55e;--red:#ef4444;
  --blue:#60a5fa;--amber:#f59e0b;--violet:#a78bfa;--cyan:#22d3ee
}
html,body{min-height:100%;background:var(--bg);color:var(--fg);font-family:'JetBrains Mono','SF Mono',ui-monospace,monospace;font-size:13px;line-height:1.45}
button,input{font:inherit}
main{max-width:1280px;margin:0 auto;padding:36px 24px 48px}
.header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:22px;padding-bottom:20px;border-bottom:1px solid var(--border)}
.header h1{font-size:22px;line-height:1.1;font-weight:700;letter-spacing:0;color:#fafafa}
.header p{margin-top:8px;color:var(--sub);font-size:12px;max-width:780px}
.meta{font-size:11px;color:var(--muted);text-align:right;white-space:nowrap}
.mono{font-variant-numeric:tabular-nums}
.grid{display:grid;gap:12px}
.kpis{grid-template-columns:repeat(auto-fit,minmax(210px,1fr));margin-bottom:22px}
.stat{background:rgba(24,24,27,.72);border:1px solid var(--border);border-radius:6px;padding:13px 14px;min-height:82px;min-width:0}
.stat.hot{border-color:#3f3f46;box-shadow:0 0 0 1px rgba(63,63,70,.35) inset}
.stat .label{font-size:10px;text-transform:uppercase;letter-spacing:.15em;color:var(--muted);line-height:1.2}
.stat .value{font-size:17px;font-weight:700;margin-top:8px;color:var(--fg);font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%}
.stat .sub{font-size:10px;color:var(--muted);margin-top:4px;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.green{color:var(--green)!important}.red{color:var(--red)!important}.blue{color:var(--blue)!important}.amber{color:var(--amber)!important}.muted{color:var(--muted)!important}
.panel{background:rgba(17,17,19,.76);border:1px solid var(--border);border-radius:6px;overflow:hidden;margin-bottom:22px}
.panel-hd{display:flex;align-items:baseline;justify-content:space-between;gap:12px;padding:14px 16px 10px;border-bottom:1px solid var(--border)}
.panel-hd h2{font-size:13px;font-weight:500;color:#e4e4e7}
.panel-hd span{font-size:11px;color:var(--muted)}
.chart{height:330px;padding:12px 14px 16px}
svg{display:block;width:100%;height:100%}
.layout{display:grid;grid-template-columns:1fr;gap:16px}
.toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.tabs{display:flex;gap:6px;flex-wrap:wrap}
.tab{border:1px solid var(--border);background:#111113;color:var(--sub);padding:5px 8px;border-radius:4px;font-size:11px;cursor:pointer}
.tab.on{border-color:#2563eb;background:rgba(37,99,235,.18);color:#bfdbfe}
input.search{background:#111113;border:1px solid var(--border);color:#e4e4e7;padding:6px 9px;border-radius:4px;font-size:11px;min-width:250px;outline:none}
input.search:focus{border-color:#52525b}
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:#141416;color:var(--muted);font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.11em;text-align:left;padding:9px 10px;border-bottom:1px solid var(--border);white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid var(--border);vertical-align:top}
th.r,td.r{text-align:right}
tbody tr:hover{background:rgba(39,39,42,.45)}
.clip{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:360px}
.small{font-size:10px;color:var(--muted)}
.pill{display:inline-block;border-radius:4px;padding:1px 6px;font-size:10px;border:1px solid var(--border);background:#18181b;color:#d4d4d8}
.empty{color:var(--muted);padding:32px 16px;text-align:center;font-size:12px}
.split{display:grid;grid-template-columns:1fr;gap:16px}
.details{display:none;background:rgba(24,24,27,.35)}
.details.open{display:table-row}
.detail-wrap{display:grid;grid-template-columns:1fr;gap:18px;padding:8px 6px}
.detail-box{border:1px solid var(--border);border-radius:5px;background:rgba(9,9,11,.35);padding:12px}
.detail-title{font-size:10px;text-transform:uppercase;letter-spacing:.14em;color:var(--muted);margin-bottom:8px}
.fill-table th{background:transparent}
.reason-card{border:1px solid var(--border);border-radius:5px;background:rgba(24,24,27,.54);padding:12px;margin:0 0 10px}
.reason-hd{display:flex;justify-content:space-between;gap:12px;font-size:11px;color:#e4e4e7;margin-bottom:8px}
.reason-row{border-left:2px solid var(--soft);padding-left:9px;margin:8px 0;color:#d4d4d8}
.reason-row .body{font-size:11px;color:var(--muted);margin-top:3px;line-height:1.5}
.scroll{max-height:430px;overflow:auto}
.warn{border:1px solid rgba(245,158,11,.35);background:rgba(245,158,11,.08);color:#fcd34d;border-radius:5px;padding:9px 11px;font-size:11px;margin-bottom:14px}
.hide{display:none!important}
@media (min-width:760px){.split{grid-template-columns:1.15fr .85fr}.detail-wrap{grid-template-columns:.95fr 1.05fr}}
@media (min-width:1120px){.layout{grid-template-columns:1fr}}
@media (max-width:720px){main{padding:22px 14px}.header{align-items:flex-start;flex-direction:column}.meta{text-align:left}.chart{height:260px}input.search{min-width:100%;width:100%}.clip{max-width:210px}}
</style>
</head>
<body>
<main>
  <div class="header">
    <div>
      <h1>Live Trading Snapshot</h1>
      <p id="subtitle">Connecting to Prophet Arena Core and the reporting API. Metrics use daily bankroll returns when P&L history is available.</p>
    </div>
    <div class="meta mono" id="hdr">connecting...</div>
  </div>

  <div id="err" class="hide warn"></div>
  <section class="grid kpis" id="kpis"></section>
  <section class="panel">
    <div class="panel-hd">
      <h2>Equity over benchmark ticks</h2>
      <span id="chartMeta">waiting for data</span>
    </div>
    <div class="chart" id="chart"></div>
  </section>

  <section class="panel">
    <div class="panel-hd">
      <h2>Leaderboard</h2>
      <span id="leaderMeta">0 participants</span>
    </div>
    <div id="leaderboard"></div>
  </section>

  <section class="panel">
    <div class="panel-hd">
      <h2>Market Ledger</h2>
      <div class="toolbar">
        <div class="tabs" id="marketTabs"></div>
        <input class="search" id="marketSearch" placeholder="Search model / market / rationale" oninput="render()" />
      </div>
    </div>
    <div id="markets"></div>
  </section>

  <section class="panel">
    <div class="panel-hd">
      <h2>Reasoning</h2>
      <span id="reasonMeta">lazy loaded</span>
    </div>
    <div id="reasoning"></div>
  </section>
</main>
<script>
const API = '/api';
const REPORT = '/report';
const REQUESTED_SLUG = __REQUESTED_SLUG__;

let exp = null, parts = [], fills = [], pnl = [], portfolios = {}, reasons = [];
let reportLeaderboard = [], reportPositions = [];
let showReasons = false, rFilter = null, marketFilter = 'all', expandedMarket = null;
let lastWarnings = [];
const COLORS = ['#22c55e','#60a5fa','#f59e0b','#a78bfa','#22d3ee','#ef4444','#84cc16','#f472b6'];
const MIN_DAILY_OBS_ELIGIBLE = 3;
const MIN_WIN_RATE_TRADES = 10;

const $ = id => document.getElementById(id);
const esc = s => { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; };
const num = v => { const n = typeof v === 'string' ? Number(v) : v; return Number.isFinite(n) ? n : null; };
const fmt = (v, d=2) => { const n=num(v); return n==null?'—':n.toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d}); };
const fmtInt = v => { const n=num(v); return n==null?'—':Math.round(n).toLocaleString(); };
const usd = v => { const n=num(v); return n==null?'—':n.toLocaleString(undefined,{style:'currency',currency:'USD',minimumFractionDigits:2,maximumFractionDigits:2}); };
const signedUsd = v => { const n=num(v); return n==null?'—':`${n>=0?'+':'-'}${usd(Math.abs(n))}`; };
const pct = (v, scale=100, d=2) => { const n=num(v); return n==null?'—':`${n>=0?'+':''}${(n*scale).toFixed(d)}%`; };
const time = iso => { if(!iso)return '—'; const d=new Date(iso); return isFinite(d)?d.toLocaleString():'—'; };
const dateKey = iso => { if(!iso)return ''; const d=new Date(iso); return isFinite(d)?d.toISOString().slice(0,10):String(iso).slice(0,10); };
const axisLabel = (ms, sameDay=false) => {
  const d = new Date(ms);
  if (!isFinite(d)) return '';
  return sameDay
    ? d.toLocaleTimeString(undefined, {hour:'2-digit', minute:'2-digit'})
    : d.toLocaleDateString(undefined, {month:'short', day:'numeric'});
};
const partLabel = p => `${p?.model||'participant'}${p?.rep!=null?`:rep${p.rep}`:''}`;

async function get(path, opts={}) {
  const r = await fetch(API + path);
  if (r.ok) return r.json();
  if (!opts.optional) throw new Error(`${path}: HTTP ${r.status}`);
  return null;
}

async function reportGet(path, opts={}) {
  const r = await fetch(REPORT + path);
  if (r.ok) return r.json();
  if (!opts.optional) throw new Error(`${path}: HTTP ${r.status}`);
  return null;
}

function asArray(payload, keys) {
  if (Array.isArray(payload)) return payload;
  for (const k of keys) if (Array.isArray(payload?.[k])) return payload[k];
  return [];
}

async function load() {
  lastWarnings = [];
  const list = await get('/experiments', {optional:true});
  let exps = asArray(list, ['experiments']);
  let reportStatusPayload = null;
  if (!exps.length && REQUESTED_SLUG) {
    reportStatusPayload = await reportGet('/status?experiment_slug='+encodeURIComponent(REQUESTED_SLUG), {optional:true});
    exps = asArray(reportStatusPayload, ['experiments']);
  }
  exps.sort((a,b) => (a.status==='RUNNING'?0:1)-(b.status==='RUNNING'?0:1) || (b.last_activity_at||'').localeCompare(a.last_activity_at||''));
  if (!exps.length) { exp = null; $('hdr').textContent = 'no experiments'; return; }
  const id = exps[0].experiment_id;
  const statusParticipants = asArray(exps[0], ['participants']);
  const [detail, p, t, prog, pnlPayload, reportLeaderboardPayload, reportPnlPayload, reportTradesPayload, reportPositionsPayload] = await Promise.all([
    get('/experiments/'+id, {optional:true}),
    get('/experiments/'+id+'/participants', {optional:true}),
    get('/experiments/'+id+'/trades?limit=1000', {optional:true}),
    get('/experiments/'+id+'/progress', {optional:true}),
    get('/experiments/'+id+'/pnl', {optional:true}),
    reportGet('/leaderboard', {optional:true}),
    reportGet('/experiments/'+id+'/pnl', {optional:true}),
    reportGet('/experiments/'+id+'/trades?limit=1000', {optional:true}),
    reportGet('/experiments/'+id+'/positions', {optional:true}),
  ]);
  exp = detail || exps[0];
  if (prog && exp) { exp._done = prog.completed||0; exp._total = prog.n_ticks||exp.n_ticks; }
  if (!prog && exp) { exp._done = exp.completed_ticks ?? exp.completed; exp._total = exp.n_ticks; }
  const participantPayload = asArray(p, ['participants']);
  parts = (participantPayload.length ? participantPayload : statusParticipants)
    .map((part, i) => ({...part, participant_idx: part.participant_idx ?? part.idx ?? i}));
  const coreFills = asArray(t, ['fills','trades']);
  const reportFills = asArray(reportTradesPayload, ['fills','trades']);
  const corePnl = asArray(pnlPayload, ['pnl']);
  const reportPnl = asArray(reportPnlPayload, ['pnl']);
  fills = reportFills.length ? reportFills : coreFills;
  pnl = reportPnl.length ? reportPnl : corePnl;
  reportLeaderboard = reportPnl.length ? asArray(reportLeaderboardPayload, ['leaderboard']) : [];
  reportPositions = asArray(reportPositionsPayload, ['positions']);

  const portfolioResults = await Promise.all(parts.map(async p => {
    const idx = p.participant_idx;
    const port = await get('/experiments/'+id+'/participants/'+idx+'/portfolio', {optional:true});
    return [idx, port];
  }));
  portfolios = Object.fromEntries(portfolioResults.filter(([, port]) => !!port));

  if (!pnl.length) lastWarnings.push('P&L history is unavailable; daily metrics need Core /experiments/{id}/pnl or the Trade UI reporting API.');
  if (!Object.keys(portfolios).length && !reportPositions.length) lastWarnings.push('Portfolio endpoint unavailable; current equity and open-position marks may be incomplete.');
  $('hdr').textContent = exp.experiment_slug || id;
}

function participantMap() {
  const m = {};
  parts.forEach(p => { m[p.participant_idx] = p; });
  return m;
}

function pnlByParticipant() {
  const by = new Map();
  for (const row of pnl) {
    const idx = Number(row.participant_idx ?? 0);
    if (!by.has(idx)) by.set(idx, []);
    by.get(idx).push(row);
  }
  for (const rows of by.values()) {
    rows.sort((a,b) => new Date(rowTimestamp(a)) - new Date(rowTimestamp(b)));
  }
  return by;
}

function rowTimestamp(row) {
  return row?.tick_ts || row?.timestamp || row?.created_at || row?.updated_at || row?.as_of || row?.date;
}

function rowEquity(row) {
  return num(row?.equity ?? row?.portfolio_value ?? row?.portfolioValue ?? row?.total_equity ?? row?.ending_equity);
}

function portfolioEquity(portfolio) {
  return num(portfolio?.equity ?? portfolio?.portfolio_value ?? portfolio?.portfolioValue ?? portfolio?.total_equity ?? portfolio?.total_value);
}

function portfolioPnl(portfolio) {
  return num(portfolio?.total_pnl ?? portfolio?.pnl ?? portfolio?.profit_loss);
}

function dailyEquity(rows) {
  const byDay = new Map();
  for (const row of rows || []) {
    const ts = rowTimestamp(row);
    const d = new Date(ts);
    const equity = rowEquity(row);
    if (!isFinite(d) || equity == null) continue;
    const key = d.toISOString().slice(0,10);
    const prev = byDay.get(key);
    if (!prev || d > prev.ts) byDay.set(key, {date:key, ts:d, equity});
  }
  return [...byDay.values()].sort((a,b)=>a.ts-b.ts);
}

function tickEquity(rows) {
  const points = [];
  for (const row of rows || []) {
    const ts = rowTimestamp(row);
    const d = new Date(ts);
    const equity = rowEquity(row);
    if (!isFinite(d) || equity == null) continue;
    points.push({ts:d, equity});
  }
  return points.sort((a,b)=>a.ts-b.ts);
}

function metricsFor(rows, startingFallback, portfolio) {
  const daily = dailyEquity(rows);
  const tickPoints = tickEquity(rows);
  const equities = daily.map(d => d.equity);
  const latestFromPnl = tickPoints.length ? tickPoints[tickPoints.length-1].equity : null;
  const latest = latestFromPnl ?? portfolioEquity(portfolio);
  const portPnl = portfolioPnl(portfolio);
  const starting = num(startingFallback) ?? (latest != null && portPnl != null ? latest - portPnl : null);
  const totalPnl = latest!=null && starting!=null ? latest - starting : portPnl;
  const ret = latest!=null && starting ? (latest - starting) / starting : null;

  const returns = [];
  for (let i=1;i<equities.length;i++) {
    if (equities[i-1] !== 0) returns.push(equities[i] / equities[i-1] - 1);
  }
  const nObs = daily.length;
  const eligible = nObs >= MIN_DAILY_OBS_ELIGIBLE;
  const mean = returns.length ? returns.reduce((s,x)=>s+x,0) / returns.length : null;
  const variance = returns.length >= 2 && mean!=null ? returns.reduce((s,x)=>s+(x-mean)**2,0)/(returns.length-1) : null;
  const std = variance!=null ? Math.sqrt(variance) : null;
  const sharpe = eligible && std != null ? (std === 0 ? 0 : (mean / std) * Math.sqrt(365)) : null;

  let cagrEstimate = null;
  if (daily.length >= 2 && daily[0].equity > 0) {
    const days = Math.max(0, (daily[daily.length-1].ts - daily[0].ts) / 86400000);
    if (days > 0) cagrEstimate = (daily[daily.length-1].equity / daily[0].equity) ** (365 / days) - 1;
  }

  let peak = equities[0] ?? null, maxDdEstimate = null;
  if (equities.length >= 2 && peak != null) {
    maxDdEstimate = 0;
    for (const e of equities) {
      if (e > peak) peak = e;
      if (peak > 0) maxDdEstimate = Math.max(maxDdEstimate, (peak - e) / peak);
    }
  }
  return {
    daily,
    tickPoints,
    nObs,
    eligible,
    starting,
    equity: latest,
    totalPnl,
    returnPct: ret,
    sharpe,
    cagr: eligible ? cagrEstimate : null,
    maxDrawdown: eligible ? maxDdEstimate : null,
  };
}

function fillCost(f) {
  const notional = num(f.notional ?? f.cost);
  if (notional != null) return Math.abs(notional);
  const shares = num(f.shares), price = num(f.price);
  return shares!=null && price!=null ? Math.abs(shares * price) : 0;
}

function fillsByPart() {
  const by = new Map();
  for (const f of fills) {
    const idx = Number(f.participant_idx ?? 0);
    if (!by.has(idx)) by.set(idx, []);
    by.get(idx).push(f);
  }
  return by;
}

function reportRowForParticipant(participant) {
  const slug = exp?.experiment_slug;
  return reportLeaderboard.find(row =>
    row.experiment_slug === slug &&
    row.model === participant.model &&
    Number(row.rep ?? 0) === Number(participant.rep ?? 0)
  ) || null;
}

function applyReportedMetrics(metrics, row) {
  if (!row) return metrics;
  const reportedObs = num(row.n_obs);
  const nObs = Math.max(metrics.nObs ?? 0, reportedObs ?? 0);
  const eligible = row.eligible ?? nObs >= MIN_DAILY_OBS_ELIGIBLE;
  const reportedReturnPct = num(row.return_pct);
  return {
    ...metrics,
    nObs,
    eligible,
    equity: metrics.equity ?? num(row.equity),
    totalPnl: metrics.totalPnl ?? num(row.total_pnl),
    returnPct: metrics.returnPct ?? (reportedReturnPct == null ? null : reportedReturnPct / 100),
    sharpe: eligible ? (metrics.sharpe ?? num(row.sharpe)) : null,
    cagr: eligible ? (metrics.cagr ?? num(row.cagr)) : null,
    maxDrawdown: eligible ? (metrics.maxDrawdown ?? num(row.max_drawdown)) : null,
  };
}

function applyReportedWinRate(winRate, row) {
  if (!row) return winRate;
  const reportedTrades = num(row.n_trades);
  const nTrades = Math.max(winRate.nTrades ?? 0, reportedTrades ?? 0);
  const reportedRate = num(row.win_rate);
  return {
    nTrades,
    winRate: reportedRate != null && nTrades >= MIN_WIN_RATE_TRADES
      ? reportedRate
      : winRate.winRate,
  };
}

function computeWinRate(partFills) {
  const groups = new Map();
  let hasOutcome = false;
  for (const f of partFills || []) {
    const outcome = f.market_outcome ?? f.outcome ?? f.resolution ?? null;
    if (outcome == null) continue;
    hasOutcome = true;
    const side = String(f.side || '').toUpperCase();
    const action = String(f.action || '').toUpperCase();
    const key = `${f.market_id}:${side}`;
    const g = groups.get(key) || {cost:0,sell:0,bought:0,sold:0,outcome,side};
    const shares = num(f.shares) ?? 0, cost = fillCost(f);
    if (action === 'BUY') { g.cost += cost; g.bought += shares; }
    if (action === 'SELL') { g.sell += cost; g.sold += shares; }
    g.outcome = outcome;
    groups.set(key,g);
  }
  if (!hasOutcome) return {nTrades:0, winRate:null};
  let wins=0,total=0;
  for (const g of groups.values()) {
    const out = String(g.outcome).toUpperCase();
    const resolvedYes = out === 'YES' || out === '1' || out === 'TRUE' || g.outcome === 1;
    const payoutPx = g.side === 'YES' ? (resolvedYes ? 1 : 0) : (resolvedYes ? 0 : 1);
    const pnl = g.sell + (g.bought - g.sold) * payoutPx - g.cost;
    total += 1;
    if (pnl > 0) wins += 1;
  }
  return {nTrades:total, winRate: total >= MIN_WIN_RATE_TRADES ? wins/total : null};
}

function leaderboardRows() {
  const ts = pnlByParticipant();
  const fb = fillsByPart();
  return parts.map((p, i) => {
    const idx = p.participant_idx;
    const reportRow = reportRowForParticipant(p);
    const m = applyReportedMetrics(
      metricsFor(ts.get(idx) || [], p.starting_cash ?? p.startingCash, portfolios[idx]),
      reportRow,
    );
    const wr = applyReportedWinRate(computeWinRate(fb.get(idx) || []), reportRow);
    return {...p, ...m, ...wr, color: COLORS[i % COLORS.length], fillCount:(fb.get(idx)||[]).length};
  }).sort((a,b)=>(num(b.totalPnl)??-Infinity)-(num(a.totalPnl)??-Infinity));
}

function currentSummary(rows) {
  const best = rows[0];
  const totalFills = fills.length;
  const markets = new Set(fills.map(f=>f.market_id)).size;
  const latestDates = rows.map(r => r.tickPoints?.at(-1)?.ts || null).filter(Boolean).sort((a,b)=>b-a);
  return {best, totalFills, markets, latest:latestDates[0] || null};
}

function render() {
  if (!exp) {
    $('kpis').innerHTML = '';
    $('leaderboard').innerHTML = '<div class="empty">No experiments found.</div>';
    $('markets').innerHTML = '';
    $('reasoning').innerHTML = '';
    return;
  }
  const rows = leaderboardRows();
  const summary = currentSummary(rows);
  const statusColor = exp.status === 'RUNNING' ? 'green' : exp.status === 'COMPLETED' ? 'blue' : exp.status === 'ABORTED' ? 'red' : '';
  $('subtitle').textContent = `${exp.experiment_slug || exp.experiment_id} · ${exp.experiment_id || ''}`;
  $('err').classList.toggle('hide', lastWarnings.length === 0);
  $('err').innerHTML = lastWarnings.map(esc).join('<br>');
  $('kpis').innerHTML = [
    stat('Status', `<span class="${statusColor}">${esc(exp.status || '—')}</span>`, `${exp._done??exp.completed_ticks??0} / ${exp._total??exp.n_ticks??'—'} ticks`),
    stat('Best model', esc(summary.best ? partLabel(summary.best) : '—'), summary.best ? `idx ${summary.best.participant_idx}` : ''),
    stat('Best P&L', signedUsd(summary.best?.totalPnl), summary.best ? pct(summary.best.returnPct,100,2) : '—', true, summary.best?.totalPnl),
    stat('Equity', usd(summary.best?.equity), summary.best ? `start ${usd(summary.best.starting)}` : ''),
    stat('Sharpe', fmt(summary.best?.sharpe,2), metricReadiness(summary.best)),
    stat('Max drawdown', summary.best?.maxDrawdown==null?'—':pct(summary.best.maxDrawdown,100,2).replace('+',''), metricReadiness(summary.best)),
    stat('CAGR', summary.best?.cagr==null?'—':pct(summary.best.cagr,100,2), metricReadiness(summary.best)),
    stat('Win rate', summary.best?.winRate==null?'—':pct(summary.best.winRate,100,1).replace('+',''), winRateReadiness(summary.best)),
  ].join('');

  renderChart(rows);
  renderLeaderboard(rows);
  renderMarkets(rows);
  renderReasoning();
}

function stat(label, value, sub='', hot=false, signedValue=null) {
  const cls = signedValue == null ? '' : signedValue >= 0 ? 'green' : 'red';
  return `<div class="stat ${hot?'hot':''}"><div class="label">${label}</div><div class="value ${cls}">${value}</div>${sub?`<div class="sub">${sub}</div>`:''}</div>`;
}

function metricReadiness(row) {
  const n = row?.nObs ?? 0;
  return row?.eligible ? `${n} daily obs` : `${n} daily obs · need ${MIN_DAILY_OBS_ELIGIBLE}`;
}

function winRateReadiness(row) {
  const n = row?.nTrades ?? 0;
  return n >= MIN_WIN_RATE_TRADES ? `${n} resolved groups` : `${n} resolved groups · need ${MIN_WIN_RATE_TRADES}`;
}

function renderChart(rows) {
  const series = rows.map((r,i) => ({
    label: partLabel(r),
    color: r.color,
    pts: (r.tickPoints || []).map(d => ({t:d.ts.getTime(), equity:d.equity})),
  })).filter(s => s.pts.length);
  const pointCount = series.reduce((n,s)=>n+s.pts.length,0);
  $('chartMeta').textContent = series.length ? `${series.length} participant curve${series.length===1?'':'s'} · ${pointCount} equity point${pointCount===1?'':'s'}` : 'P&L history unavailable';
  if (!series.length) {
    $('chart').innerHTML = '<div class="empty">P&L history is unavailable from Core and the reporting API. Current equity is shown in the KPI cards.</div>';
    return;
  }
  const all = series.flatMap(s=>s.pts);
  const minT = Math.min(...all.map(p=>p.t)), maxT = Math.max(...all.map(p=>p.t));
  let minY = Math.min(...all.map(p=>p.equity)), maxY = Math.max(...all.map(p=>p.equity));
  if (minY === maxY) { minY -= Math.max(1,minY*.02); maxY += Math.max(1,maxY*.02); }
  const padY = (maxY-minY)*.12; minY -= padY; maxY += padY;
  const W=1000,H=300,L=58,R=18,T=16,B=36;
  const singleTimestamp = minT === maxT;
  const sameDay = dateKey(new Date(minT).toISOString()) === dateKey(new Date(maxT).toISOString());
  const x = t => singleTimestamp ? L + (W-L-R)/2 : L + ((t-minT)/(maxT-minT))*(W-L-R);
  const y = v => T + (1-(v-minY)/(maxY-minY))*(H-T-B);
  const lines = series.map(s => {
    const d = s.pts.map((p,i)=>`${i?'L':'M'}${x(p.t).toFixed(1)},${y(p.equity).toFixed(1)}`).join(' ');
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>`;
  }).join('');
  const dots = series.map(s => s.pts.length === 1 ? `<circle cx="${x(s.pts[0].t).toFixed(1)}" cy="${y(s.pts[0].equity).toFixed(1)}" r="4" fill="${s.color}"/>` : '').join('');
  const yTicks = [minY, (minY+maxY)/2, maxY];
  const grid = yTicks.map(v=>`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#27272a"/><text x="8" y="${y(v)+4}" fill="#a1a1aa" font-size="12">${usd(v)}</text>`).join('');
  const legend = series.map((s,i)=>`<span style="color:${s.color};margin-right:14px">${esc(s.label)}</span>`).join('');
  const rightLabel = singleTimestamp ? '' : `<text x="${W-R-92}" y="${H-8}" fill="#a1a1aa" font-size="12">${axisLabel(maxT, sameDay)}</text>`;
  $('chart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${grid}<line x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}" stroke="#3f3f46"/>${lines}${dots}<text x="${L}" y="${H-8}" fill="#a1a1aa" font-size="12">${axisLabel(minT, sameDay)}</text>${rightLabel}</svg><div class="small mono" style="margin-top:-8px">${legend}</div>`;
}

function renderLeaderboard(rows) {
  $('leaderMeta').textContent = `${rows.length} participant${rows.length===1?'':'s'}`;
  if (!rows.length) { $('leaderboard').innerHTML = '<div class="empty">No participants yet.</div>'; return; }
  $('leaderboard').innerHTML = `<div class="scroll"><table>
    <thead><tr><th class="r">Rank</th><th>Model</th><th class="r">Equity</th><th class="r">P&L</th><th class="r">Return</th><th class="r">CAGR</th><th class="r">Sharpe</th><th class="r">MDD</th><th class="r">Win Rate</th><th class="r">Fills</th><th class="r">Daily Obs</th></tr></thead>
    <tbody>${rows.map((r,i)=>`
      <tr>
        <td class="r muted">${i+1}</td>
        <td><span style="color:${r.color}">■</span> ${esc(partLabel(r))}<div class="small">idx ${r.participant_idx}</div></td>
        <td class="r mono">${usd(r.equity)}</td>
        <td class="r mono ${r.totalPnl>=0?'green':'red'}">${signedUsd(r.totalPnl)}</td>
        <td class="r mono ${r.returnPct>=0?'green':'red'}">${pct(r.returnPct,100,2)}</td>
        <td class="r mono">${r.cagr==null ? '—' : pct(r.cagr,100,2)}</td>
        <td class="r mono">${r.sharpe==null ? '—' : fmt(r.sharpe,2)}</td>
        <td class="r mono">${r.maxDrawdown!=null ? pct(r.maxDrawdown,100,2).replace('+','') : '—'}</td>
        <td class="r mono">${r.winRate==null?'—':pct(r.winRate,100,1).replace('+','')}</td>
        <td class="r mono">${fmtInt(r.fillCount)}</td>
        <td class="r mono">${fmtInt(r.nObs)}</td>
      </tr>`).join('')}</tbody>
  </table></div>`;
}

function positionRows() {
  const rowsByKey = new Map();
  for (const pos of reportPositions || []) {
    const idx = Number(pos.participant_idx ?? 0);
    const side = String(pos.side || '').toUpperCase();
    rowsByKey.set(`${idx}:${pos.market_id}:${side}`, {...pos, participant_idx:idx});
  }
  for (const [idx, port] of Object.entries(portfolios)) {
    for (const pos of port.positions || []) {
      const side = String(pos.side || '').toUpperCase();
      rowsByKey.set(`${Number(idx)}:${pos.market_id}:${side}`, {...pos, participant_idx:Number(idx)});
    }
  }
  return [...rowsByKey.values()];
}

function marketRows() {
  const pmap = participantMap();
  const pos = positionRows();
  const posByKey = new Map(pos.map(p => [`${p.participant_idx}:${p.market_id}:${String(p.side).toUpperCase()}`, p]));
  const groups = new Map();
  for (const f of fills) {
    const idx = Number(f.participant_idx ?? 0);
    const mid = f.market_id || 'unknown';
    const key = `${idx}:${mid}`;
    const g = groups.get(key) || {participant_idx:idx, market_id:mid, question:f.market_question || f.question || mid, fills:[], buy:0, sell:0, netCost:0, netYes:0, netNo:0, pnl:null, outcome:f.market_outcome ?? f.outcome ?? null};
    const action = String(f.action||'').toUpperCase(), side = String(f.side||'').toUpperCase();
    const shares = num(f.shares) ?? 0, cost = fillCost(f);
    if (action === 'BUY') { g.buy += 1; g.netCost += cost; if (side==='YES') g.netYes += shares; else g.netNo += shares; }
    if (action === 'SELL') { g.sell += 1; g.netCost -= cost; if (side==='YES') g.netYes -= shares; else g.netNo -= shares; }
    if (f.market_question || f.question) g.question = f.market_question || f.question;
    g.fills.push(f);
    groups.set(key,g);
  }
  for (const g of groups.values()) {
    let pnlVal = 0, saw = false;
    for (const side of ['YES','NO']) {
      const p = posByKey.get(`${g.participant_idx}:${g.market_id}:${side}`);
      if (p) {
        const u = num(p.unrealized_pnl), r = num(p.realized_pnl);
        if (u!=null || r!=null) { pnlVal += (u??0) + (r??0); saw = true; }
      }
    }
    if (saw) g.pnl = pnlVal;
    g.model = partLabel(pmap[g.participant_idx]);
    g.last = g.fills.reduce((m,f)=>!m || new Date(f.filled_at||f.timestamp) > new Date(m) ? (f.filled_at||f.timestamp) : m, null);
  }
  return [...groups.values()].sort((a,b)=>(num(b.pnl)??-Math.abs(b.netCost))-(num(a.pnl)??-Math.abs(a.netCost)));
}

function renderMarkets() {
  const tabs = ['all','open','flat','won','lost'];
  $('marketTabs').innerHTML = tabs.map(t=>`<button class="tab ${marketFilter===t?'on':''}" onclick="marketFilter='${t}';render()">${t}</button>`).join('');
  const q = $('marketSearch')?.value?.trim()?.toLowerCase() || '';
  let rows = marketRows();
  rows = rows.filter(r => {
    const flat = Math.abs(r.netYes) < 1e-9 && Math.abs(r.netNo) < 1e-9;
    if (marketFilter === 'open' && flat) return false;
    if (marketFilter === 'flat' && !flat) return false;
    if (marketFilter === 'won' && !(r.pnl != null && r.pnl > 0)) return false;
    if (marketFilter === 'lost' && !(r.pnl != null && r.pnl < 0)) return false;
    if (!q) return true;
    return `${r.model} ${r.market_id} ${r.question} ${r.fills.map(f=>f.rationale||'').join(' ')}`.toLowerCase().includes(q);
  });
  if (!rows.length) { $('markets').innerHTML = '<div class="empty">No markets match the current filters.</div>'; return; }
  $('markets').innerHTML = `<div class="scroll"><table>
    <thead><tr><th></th><th>Last Activity</th><th>Model</th><th>Market</th><th class="r">Fills</th><th class="r">Net Qty</th><th class="r">Net Cost</th><th class="r">P&L</th></tr></thead>
    <tbody>${rows.map((r,i)=>marketRow(r,i)).join('')}</tbody>
  </table></div>`;
}

function marketRow(r, i) {
  const key = `${r.participant_idx}:${r.market_id}`;
  const open = expandedMarket === key;
  const netQty = [r.netYes?`${fmtInt(r.netYes)} YES`:null, r.netNo?`${fmtInt(r.netNo)} NO`:null].filter(Boolean).join(' / ') || 'flat';
  const rationale = latestRationale(r.fills);
  return `<tr onclick="expandedMarket=expandedMarket==='${esc(key)}'?null:'${esc(key)}';render()" style="cursor:pointer">
    <td class="muted">${open?'▾':'▸'}</td><td class="mono muted">${time(r.last)}</td><td>${esc(r.model)}</td>
    <td><div class="clip" title="${esc(r.question)}">${esc(r.question)}</div><div class="small">${esc(r.market_id)}</div></td>
    <td class="r mono">${r.fills.length}<span class="small"> (${r.buy}B${r.sell?` / ${r.sell}S`:''})</span></td>
    <td class="r mono">${esc(netQty)}</td><td class="r mono">${usd(r.netCost)}</td>
    <td class="r mono ${r.pnl==null?'muted':r.pnl>=0?'green':'red'}">${r.pnl==null?'—':signedUsd(r.pnl)}</td>
  </tr><tr class="details ${open?'open':''}"><td colspan="8"><div class="detail-wrap">
    <div class="detail-box"><div class="detail-title">P&L breakdown</div>
      <div class="small">Current row P&L comes from the latest portfolio position mark when available. Flat/settled markets may need Core settlement history to reconcile perfectly.</div>
      <table class="fill-table" style="margin-top:10px"><tbody>
        <tr><td>Net cash flow</td><td class="r mono">${signedUsd(-r.netCost)}</td></tr>
        <tr><td>Open inventory</td><td class="r mono">${esc(netQty)}</td></tr>
        <tr><td>Marked P&L</td><td class="r mono ${r.pnl==null?'muted':r.pnl>=0?'green':'red'}">${r.pnl==null?'—':signedUsd(r.pnl)}</td></tr>
      </tbody></table>
    </div>
    <div class="detail-box"><div class="detail-title">Fills</div>
      <table class="fill-table"><thead><tr><th>Time</th><th>Action</th><th>Side</th><th class="r">Shares</th><th class="r">Price</th><th class="r">Notional</th></tr></thead>
      <tbody>${r.fills.slice().sort((a,b)=>new Date(a.filled_at||a.timestamp)-new Date(b.filled_at||b.timestamp)).map(f=>`<tr><td class="mono muted">${time(f.filled_at||f.timestamp)}</td><td class="${String(f.action).toUpperCase()==='BUY'?'green':'red'}">${esc(f.action||'')}</td><td>${esc(f.side||'')}</td><td class="r mono">${fmt(f.shares,2)}</td><td class="r mono">${fmt(f.price,4)}</td><td class="r mono">${usd(fillCost(f))}</td></tr>`).join('')}</tbody></table>
      ${rationale?`<div class="detail-title" style="margin-top:12px">Latest rationale</div><div class="small" style="color:#d4d4d8">${esc(rationale)}</div>`:''}
    </div>
  </div></td></tr>`;
}

function latestRationale(fs) {
  const withReason = fs.filter(f => f.rationale || f.reasoning).sort((a,b)=>new Date(b.filled_at||b.timestamp)-new Date(a.filled_at||a.timestamp));
  return withReason[0]?.rationale || withReason[0]?.reasoning || '';
}

function renderReasoning() {
  const pmap = participantMap();
  const entries = rFilter===null ? reasons : reasons.filter(r=>r.participant_idx===rFilter);
  const controls = `<div class="toolbar" style="padding:12px 16px">${parts.length>1?`<div class="tabs"><button class="tab ${rFilter===null?'on':''}" onclick="rFilter=null;render()">all</button>${parts.map(p=>`<button class="tab ${rFilter===p.participant_idx?'on':''}" onclick="rFilter=${p.participant_idx};render()">${esc(partLabel(p))}</button>`).join('')}</div>`:''}<button class="tab ${showReasons?'on':''}" onclick="toggleReasons()">${showReasons?'hide':'load'} reasoning</button></div>`;
  $('reasonMeta').textContent = reasons.length ? `${reasons.length} entries` : 'lazy loaded';
  if (!showReasons) { $('reasoning').innerHTML = controls; return; }
  if (!entries.length) { $('reasoning').innerHTML = controls + '<div class="empty">No reasoning data.</div>'; return; }
  $('reasoning').innerHTML = controls + '<div class="scroll" style="padding:0 16px 16px">' + renderReasons(entries, pmap) + '</div>';
}

function renderReasons(entries, pm) {
  let fb = '';
  if (!entries.length) return fb + '<div class="empty">No reasoning data</div>';
  return fb + entries.map(e => {
    const r = e.reasoning||{}, model = pm[e.participant_idx] ? partLabel(pm[e.participant_idx]) : '#'+e.participant_idx;
    let body = '';
    if (r.forecasts) {
      body += '<div class="detail-title blue">Forecasts</div>';
      for (const [mid,f] of Object.entries(r.forecasts)) {
        const py = f.p_yes!=null ? `<span class="pill" style="background:rgba(100,100,255,.15);margin-left:6px">P(YES) = ${(f.p_yes*100).toFixed(1)}%</span>` : '';
        body += `<div class="reason-row"><div>${esc(f.question||mid)}${py}</div>${f.rationale?'<div class="body">'+esc(f.rationale)+'</div>':''}</div>`;
      }
    }
    if (r.decisions) {
      body += '<div class="detail-title green" style="margin-top:10px">Decisions</div>';
      for (const [mid,d] of Object.entries(r.decisions)) {
        const hold=d.recommendation==='HOLD', buy=(d.recommendation||'').includes('BUY');
        const c=hold?'var(--dim)':buy?'var(--green)':'var(--red)';
        const bg=hold?'rgba(100,100,100,.15)':buy?'rgba(34,197,94,.15)':'rgba(239,68,68,.15)';
        const sz=d.size_usd!=null&&d.size_usd>0?`<span style="margin-left:6px;font-size:11px;color:var(--dim)">${usd(d.size_usd)}</span>`:'';
        body += `<div class="reason-row"><div>${esc(d.question||mid)}<span class="pill" style="background:${bg};color:${c};margin-left:6px">${d.recommendation||'-'}</span>${sz}</div>${d.rationale?'<div class="body">'+esc(d.rationale)+'</div>':''}</div>`;
      }
    }
    return `<div class="reason-card"><div class="reason-hd"><span>${esc(model)}</span><span class="mono muted">${time(e.tick_id)}</span></div>${body || '<div class="small">No structured reasoning in this entry.</div>'}</div>`;
  }).join('');
}

async function toggleReasons() {
  showReasons = !showReasons;
  if (showReasons && !reasons.length && exp) {
    const d = await get('/experiments/'+exp.experiment_id+'/reasoning?limit=200', {optional:true});
    reasons = asArray(d, ['reasoning']);
  }
  render();
}

(async () => {
  try { await load(); } catch(e) { $('err').textContent='Error: '+e.message; $('err').classList.remove('hide'); }
  render();
  setInterval(async () => { try { await load(); render(); } catch(e) {} }, 10000);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
