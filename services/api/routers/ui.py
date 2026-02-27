"""
Sentinel — Embedded Analyst UI Router

Serves a premium analyst interface as a single HTML page.
No Node.js, no build step, no external dependencies.
Works with the /api/v1/* endpoints.
"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def analyst_ui():
    """Serve the embedded analyst UI."""
    return ANALYST_UI_HTML


ANALYST_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="description" content="Sentinel — Multi-modal spatiotemporal entity reasoning engine">
<title>Sentinel — Entity Reasoning Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg-primary:#0a0e17;--bg-secondary:#111827;--bg-tertiary:#1a2235;--bg-card:#151d2e;--bg-card-hover:#1c2740;--bg-elevated:#1e293b;--surface-glass:rgba(17,24,39,.75);--border-subtle:rgba(148,163,184,.1);--border-default:rgba(148,163,184,.15);--border-strong:rgba(148,163,184,.25);--text-primary:#f1f5f9;--text-secondary:#94a3b8;--text-tertiary:#64748b;--text-muted:#475569;--accent-blue:#3b82f6;--accent-blue-glow:rgba(59,130,246,.15);--accent-cyan:#06b6d4;--accent-cyan-glow:rgba(6,182,212,.15);--accent-emerald:#10b981;--accent-emerald-glow:rgba(16,185,129,.15);--accent-amber:#f59e0b;--accent-amber-glow:rgba(245,158,11,.15);--accent-red:#ef4444;--accent-red-glow:rgba(239,68,68,.15);--accent-violet:#8b5cf6;--accent-violet-glow:rgba(139,92,246,.15);--layer-raw:#3b82f6;--layer-derived:#8b5cf6;--layer-inference:#06b6d4;--font-sans:'Inter',-apple-system,sans-serif;--font-mono:'JetBrains Mono',monospace;--radius-sm:6px;--radius-md:10px;--radius-lg:14px;--radius-full:9999px;--transition-fast:150ms cubic-bezier(.4,0,.2,1);--transition-base:250ms cubic-bezier(.4,0,.2,1)}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:16px;-webkit-font-smoothing:antialiased}
body{font-family:var(--font-sans);background:var(--bg-primary);color:var(--text-primary);line-height:1.6;min-height:100vh}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--border-strong);border-radius:var(--radius-full)}
.app{display:grid;grid-template-rows:auto 1fr;min-height:100vh}
.header{display:flex;align-items:center;justify-content:space-between;padding:.75rem 1.5rem;background:var(--surface-glass);backdrop-filter:blur(20px);border-bottom:1px solid var(--border-subtle);position:sticky;top:0;z-index:100}
.header-brand{display:flex;align-items:center;gap:.75rem}
.header-logo{width:32px;height:32px;background:linear-gradient(135deg,var(--accent-cyan),var(--accent-blue));border-radius:var(--radius-sm);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.8125rem;color:#fff;box-shadow:0 0 20px rgba(6,182,212,.2)}
.header-title{font-size:1.125rem;font-weight:700;letter-spacing:-.02em;background:linear-gradient(135deg,var(--text-primary),var(--accent-cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.header-subtitle{font-size:.6875rem;color:var(--text-tertiary);font-family:var(--font-mono);text-transform:uppercase;letter-spacing:.1em}
.header-nav{display:flex;gap:4px}
.nav-btn{padding:.5rem 1rem;background:transparent;border:1px solid transparent;border-radius:var(--radius-sm);color:var(--text-secondary);font-family:var(--font-sans);font-size:.8125rem;font-weight:500;cursor:pointer;transition:all var(--transition-fast);position:relative}
.nav-btn:hover{background:var(--bg-card);color:var(--text-primary);border-color:var(--border-default)}
.nav-btn.active{background:var(--accent-blue-glow);color:var(--accent-blue);border-color:rgba(59,130,246,.3)}
.nav-btn.active::after{content:'';position:absolute;bottom:-1px;left:20%;right:20%;height:2px;background:var(--accent-blue);border-radius:var(--radius-full)}
.header-status{display:flex;align-items:center;gap:1rem}
.status-indicator{display:flex;align-items:center;gap:.5rem;font-size:.6875rem;font-family:var(--font-mono);color:var(--text-tertiary)}
.status-dot{width:6px;height:6px;border-radius:50%;animation:pulse 2s infinite}
.status-dot.on{background:var(--accent-emerald);box-shadow:0 0 6px var(--accent-emerald)}
.status-dot.off{background:var(--accent-red);box-shadow:0 0 6px var(--accent-red)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.main{padding:1.5rem;display:flex;flex-direction:column;gap:1.5rem}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
.grid-wide{display:grid;grid-template-columns:2fr 1fr;gap:1.5rem}
.stat-card{background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:var(--radius-lg);padding:1.25rem;transition:all var(--transition-base);position:relative;overflow:hidden}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;opacity:0;transition:opacity var(--transition-base)}
.stat-card:hover{border-color:var(--border-default);transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,.4)}
.stat-card:hover::before{opacity:1}
.stat-card.blue::before{background:linear-gradient(90deg,var(--accent-blue),var(--accent-cyan))}
.stat-card.violet::before{background:linear-gradient(90deg,var(--accent-violet),var(--accent-blue))}
.stat-card.emerald::before{background:linear-gradient(90deg,var(--accent-emerald),var(--accent-cyan))}
.stat-card.amber::before{background:linear-gradient(90deg,var(--accent-amber),var(--accent-red))}
.stat-label{font-size:.6875rem;color:var(--text-tertiary);text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:.5rem}
.stat-value{font-size:1.75rem;font-weight:800;letter-spacing:-.02em;line-height:1;margin-bottom:.25rem}
.stat-detail{font-size:.6875rem;color:var(--text-tertiary);font-family:var(--font-mono)}
.panel{background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:var(--radius-lg);overflow:hidden}
.panel-header{display:flex;align-items:center;justify-content:space-between;padding:1rem 1.25rem;border-bottom:1px solid var(--border-subtle)}
.panel-title{font-size:.9375rem;font-weight:700;letter-spacing:-.01em}
.panel-body{padding:1rem 1.25rem}
.timeline{position:relative;padding-left:2rem}
.timeline::before{content:'';position:absolute;left:15px;top:0;bottom:0;width:2px;background:linear-gradient(180deg,var(--layer-raw),var(--layer-derived),var(--layer-inference));opacity:.3}
.tl-entry{position:relative;padding:1rem;margin-bottom:.75rem;background:var(--bg-tertiary);border:1px solid var(--border-subtle);border-radius:var(--radius-md);transition:all var(--transition-fast);cursor:pointer}
.tl-entry:hover{border-color:var(--border-default);background:var(--bg-card-hover);transform:translateX(4px)}
.tl-entry::before{content:'';position:absolute;left:calc(-2rem + 10px);top:50%;transform:translateY(-50%);width:12px;height:12px;border-radius:50%;border:2px solid;background:var(--bg-primary)}
.tl-entry.raw::before{border-color:var(--layer-raw);box-shadow:0 0 8px var(--layer-raw)}
.tl-entry.derived::before{border-color:var(--layer-derived);box-shadow:0 0 8px var(--layer-derived)}
.tl-entry.inference::before{border-color:var(--layer-inference);box-shadow:0 0 8px var(--layer-inference)}
.tl-meta{display:flex;align-items:center;gap:.75rem;margin-bottom:.5rem;flex-wrap:wrap}
.tl-time{font-family:var(--font-mono);font-size:.6875rem;color:var(--text-secondary);font-weight:500}
.tl-src{font-size:.6875rem;padding:2px 8px;border-radius:var(--radius-full);background:var(--bg-elevated);color:var(--text-tertiary);font-family:var(--font-mono)}
.tag{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:var(--radius-full);font-size:.6875rem;font-weight:600;font-family:var(--font-mono);letter-spacing:.02em}
.tag.raw{background:rgba(59,130,246,.12);color:var(--layer-raw)}
.tag.derived{background:rgba(139,92,246,.12);color:var(--layer-derived)}
.tag.inference{background:rgba(6,182,212,.12);color:var(--layer-inference)}
.tag.high{background:var(--accent-emerald-glow);color:var(--accent-emerald)}
.tag.med{background:var(--accent-amber-glow);color:var(--accent-amber)}
.tag.low{background:var(--accent-red-glow);color:var(--accent-red)}
.tag.model{background:rgba(148,163,184,.06);color:var(--text-tertiary)}
.tl-summary{font-size:.8125rem;color:var(--text-primary);line-height:1.5}
.tl-tags{display:flex;gap:.5rem;margin-top:.5rem;flex-wrap:wrap}
.conf-bar{height:4px;background:var(--bg-primary);border-radius:var(--radius-full);overflow:hidden;margin-top:.5rem}
.conf-fill{height:100%;border-radius:var(--radius-full);transition:width .4s}
.conf-fill.high{background:linear-gradient(90deg,var(--accent-emerald),var(--accent-cyan))}
.conf-fill.med{background:linear-gradient(90deg,var(--accent-amber),var(--accent-emerald))}
.conf-fill.low{background:linear-gradient(90deg,var(--accent-red),var(--accent-amber))}
.filter-group{display:flex;gap:.5rem;flex-wrap:wrap}
.filter-pill{padding:4px 12px;border-radius:var(--radius-full);font-size:.6875rem;font-weight:500;border:1px solid var(--border-default);background:transparent;color:var(--text-secondary);cursor:pointer;transition:all var(--transition-fast);font-family:var(--font-sans)}
.filter-pill:hover{background:var(--bg-card);color:var(--text-primary)}
.filter-pill.active{background:var(--accent-blue-glow);color:var(--accent-blue);border-color:rgba(59,130,246,.3)}
.btn{display:inline-flex;align-items:center;gap:.5rem;padding:.5rem 1rem;border:1px solid var(--border-default);border-radius:var(--radius-sm);font-family:var(--font-sans);font-size:.6875rem;font-weight:600;cursor:pointer;transition:all var(--transition-fast);background:var(--bg-tertiary);color:var(--text-secondary)}
.btn:hover{background:var(--bg-card-hover);color:var(--text-primary)}
.btn-primary{background:var(--accent-blue);color:#fff;border-color:var(--accent-blue)}
.btn-primary:hover{background:#2563eb;box-shadow:0 0 20px rgba(59,130,246,.2)}
.graph-box{position:relative;background:var(--bg-tertiary);border-radius:var(--radius-md);min-height:420px;overflow:hidden}
.graph-box svg{width:100%;height:420px}
.graph-legend{display:flex;gap:1rem;padding:.75rem 1rem;border-top:1px solid var(--border-subtle)}
.legend-item{display:flex;align-items:center;gap:.5rem;font-size:.6875rem;color:var(--text-tertiary)}
.legend-dot{width:10px;height:10px;border-radius:50%}
.hyp-card{background:var(--bg-tertiary);border:1px solid var(--border-subtle);border-radius:var(--radius-md);padding:1rem;margin-bottom:.75rem;transition:all var(--transition-fast)}
.hyp-card:hover{border-color:var(--accent-cyan);box-shadow:0 0 12px rgba(6,182,212,.1)}
.scores-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem}
.score-item{text-align:center;padding:.5rem;background:var(--bg-primary);border-radius:var(--radius-sm)}
.score-label{font-size:.6875rem;color:var(--text-tertiary);display:block;margin-bottom:2px}
.score-val{font-size:.9375rem;font-weight:700;font-family:var(--font-mono)}
.health-item{display:flex;justify-content:space-between;align-items:center;padding:.75rem;background:var(--bg-tertiary);border-radius:var(--radius-sm);border:1px solid var(--border-subtle);margin-bottom:.5rem}
.health-left{display:flex;align-items:center;gap:.75rem}
.health-name{font-size:.8125rem;font-weight:600}
.health-detail{font-size:.6875rem;color:var(--text-tertiary);font-family:var(--font-mono)}
.model-info{margin-top:1rem;padding:.75rem;background:var(--bg-primary);border-radius:var(--radius-sm);border:1px solid var(--border-subtle)}
.model-info-title{font-size:.6875rem;color:var(--text-tertiary);margin-bottom:.5rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em}
.flow{display:flex;gap:.5rem;align-items:center}
.flow-arrow{color:var(--text-muted)}
.audit-table{width:100%;border-collapse:separate;border-spacing:0}
.audit-table th{padding:.5rem .75rem;text-align:left;font-size:.6875rem;font-weight:600;color:var(--text-tertiary);text-transform:uppercase;letter-spacing:.08em;border-bottom:1px solid var(--border-default)}
.audit-table td{padding:.5rem .75rem;font-size:.6875rem;color:var(--text-secondary);border-bottom:1px solid var(--border-subtle);font-family:var(--font-mono)}
.audit-table tr:hover td{background:var(--bg-tertiary);color:var(--text-primary)}
.audit-footer{margin-top:1rem;padding:.75rem;background:var(--bg-primary);border-radius:var(--radius-sm);border:1px solid var(--border-subtle);display:flex;justify-content:space-between;font-size:.6875rem;color:var(--text-tertiary);font-family:var(--font-mono)}
.pipeline-stage{padding:1rem;margin-bottom:.75rem;background:var(--bg-tertiary);border:1px solid var(--border-subtle);border-radius:var(--radius-md);border-left:3px solid}
.pipeline-stage.active{background:rgba(16,185,129,.05);border-color:rgba(16,185,129,.2)}
.pipeline-stage .stage-title{font-size:.8125rem;font-weight:600;margin-bottom:4px}
.pipeline-stage .stage-desc{font-size:.6875rem;color:var(--text-secondary)}
.disclaimer{padding:.75rem;background:var(--bg-primary);border-radius:var(--radius-sm);border:1px dashed var(--border-default);text-align:center;color:var(--text-tertiary);font-size:.6875rem}
.hidden{display:none!important}
@media(max-width:1200px){.grid-4{grid-template-columns:repeat(2,1fr)}.grid-2,.grid-wide{grid-template-columns:1fr}}
@media(max-width:768px){.grid-4{grid-template-columns:1fr}.header{flex-wrap:wrap;gap:.75rem}.header-nav{order:3;width:100%;overflow-x:auto}.main{padding:.75rem}}
</style>
</head>
<body>
<div class="app">
<header class="header">
  <div class="header-brand">
    <div class="header-logo">S</div>
    <div>
      <div class="header-title">Sentinel</div>
      <div class="header-subtitle">Spatiotemporal Entity Reasoning Engine</div>
    </div>
  </div>
  <nav class="header-nav" id="nav"></nav>
  <div class="header-status" id="status-bar"></div>
</header>
<main class="main" id="content"></main>
</div>

<script>
const VIEWS = ['dashboard','timeline','graph','hypotheses','audit'];
let currentView = 'dashboard';
let layerFilter = null;
let selectedNode = null;

// ── DATA ──
const TIMELINE = [
  {id:'1',layer:'raw',type:'frame',ts:'2026-02-26T10:15:00Z',src:'cam-001',conf:1.0,sum:'Raw frame captured at Main St & 5th Ave — 1920×1080, SHA256 verified',mv:null},
  {id:'2',layer:'derived',type:'detection',ts:'2026-02-26T10:15:00Z',src:'cam-001',conf:.92,sum:'Person detected (92%) + white sedan (88%) — 3 detections',mv:'yolov8n-8.3.0'},
  {id:'3',layer:'derived',type:'plate_read',ts:'2026-02-26T10:17:30Z',src:'cam-001',conf:.87,sum:'License plate ABC-1234 recognized on white sedan',mv:'paddleocr-2.8.0'},
  {id:'4',layer:'derived',type:'face_embedding',ts:'2026-02-26T10:22:00Z',src:'cam-002',conf:.78,sum:'Face match (cosine sim 0.78) with subject from cam-001',mv:'insightface-0.7.3'},
  {id:'5',layer:'raw',type:'phone_ping',ts:'2026-02-26T10:25:00Z',src:'tower-42',conf:1.0,sum:'Phone ping detected — 150m accuracy radius, -72 dBm',mv:null},
  {id:'6',layer:'derived',type:'plate_read',ts:'2026-02-26T10:45:00Z',src:'cam-003',conf:.94,sum:'Plate ABC-1234 on I-95 Mile 42 — exact match confirmed',mv:'paddleocr-2.8.0'},
  {id:'7',layer:'inference',type:'constraint',ts:'2026-02-26T10:50:00Z',src:'reasoner',conf:.85,sum:'Speed check PASSED: 28 min, ~18 mi = ~39 mph (< 80 mph limit)',mv:'reasoner-det-1.0'},
  {id:'8',layer:'derived',type:'face_embedding',ts:'2026-02-26T11:05:00Z',src:'cam-002',conf:.45,sum:'Possible face match (0.45) — BELOW threshold, flagged for review',mv:'insightface-0.7.3'},
];

const NODES = [
  {id:'p1',label:'Unknown Subject A',type:'Person',mv:'insightface-0.7.3',conf:.72,evts:['de-1','de-3']},
  {id:'v1',label:'Sedan ABC-1234',type:'Vehicle',mv:'paddleocr-2.8.0',conf:.91,evts:['de-2','de-5']},
  {id:'c1',label:'Main St',type:'Camera',mv:null,conf:1,evts:[]},
  {id:'c2',label:'Park N',type:'Camera',mv:null,conf:1,evts:[]},
  {id:'c3',label:'I-95 M42',type:'Camera',mv:null,conf:1,evts:[]},
  {id:'pl1',label:'CBD',type:'Place',mv:null,conf:1,evts:[]},
  {id:'ph1',label:'Phone',type:'Phone',mv:null,conf:.65,evts:['re-4']},
];

const EDGES = [
  {s:'p1',t:'c1',rel:'APPEARED_IN',conf:.92,reason:'YOLOv8+InsightFace',model:'insightface-0.7.3'},
  {s:'p1',t:'c2',rel:'APPEARED_IN',conf:.78,reason:'Face cosine 0.78',model:'insightface-0.7.3'},
  {s:'v1',t:'c1',rel:'APPEARED_IN',conf:.87,reason:'PaddleOCR plate',model:'paddleocr-2.8.0'},
  {s:'v1',t:'c3',rel:'APPEARED_IN',conf:.94,reason:'Plate exact match',model:'paddleocr-2.8.0'},
  {s:'p1',t:'v1',rel:'NEAR',conf:.75,reason:'Co-located 150s window',model:'reasoner-1.0'},
  {s:'ph1',t:'p1',rel:'NEAR',conf:.55,reason:'150m + 3min overlap',model:'reasoner-1.0'},
  {s:'c1',t:'pl1',rel:'LOCATED_AT',conf:1,reason:'Manual',model:'manual'},
];

const HYP = {title:'Subject A: CBD → Park → Highway',desc:'Face match at Main St (10:15) + Park (10:22), vehicle ABC-1234 on I-95 (10:45). Travel consistent.',score:.73,stage:'deterministic',status:'active',
  scores:{'Temporal':.85,'Spatial':.90,'Face Avg':.72,'LPR Match':.91,'Phone':.55,'Combined':.73}};

const AUDIT = [
  {id:1,action:'create',actor:'system',res:'camera',mv:null,detail:'3 cameras init',ts:'09:00'},
  {id:2,action:'inference',actor:'detector-v1',res:'detection',mv:'yolov8n-8.3.0',detail:'45→12 dets',ts:'10:15'},
  {id:3,action:'inference',actor:'face-embed',res:'face_match',mv:'insightface-0.7.3',detail:'2 matches',ts:'10:20'},
  {id:4,action:'inference',actor:'lpr-engine',res:'plate_read',mv:'paddleocr-2.8.0',detail:'ABC-1234',ts:'10:17'},
  {id:5,action:'create',actor:'analyst',res:'hypothesis',mv:null,detail:'Travel hyp',ts:'11:30'},
  {id:6,action:'inference',actor:'reasoner',res:'constraint',mv:'det-1.0',detail:'Speed pass',ts:'10:50'},
];

const NCOLORS = {Person:'#3b82f6',Vehicle:'#f59e0b',Camera:'#10b981',Place:'#8b5cf6',Phone:'#06b6d4'};

// ── HELPERS ──
function confClass(c){return c>=.75?'high':c>=.5?'med':'low'}
function confColor(c){return c>=.75?'var(--accent-emerald)':c>=.5?'var(--accent-amber)':'var(--accent-red)'}
function fmtTime(ts){return new Date(ts).toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit'})}
function h(tag,cls,inner,attrs=''){return `<${tag} class="${cls}" ${attrs}>${inner}</${tag}>`}

// ── RENDER ──
function render(){
  // Nav
  document.getElementById('nav').innerHTML = VIEWS.map(v=>
    `<button class="nav-btn ${currentView===v?'active':''}" onclick="setView('${v}')">${v[0].toUpperCase()+v.slice(1)}</button>`
  ).join('');

  // Status
  document.getElementById('status-bar').innerHTML = ['API','PostGIS','Neo4j','Redis'].map(s=>
    `<div class="status-indicator"><div class="status-dot on"></div>${s}</div>`
  ).join('');

  const el = document.getElementById('content');
  el.innerHTML = '';

  if(currentView==='dashboard') renderDashboard(el);
  else if(currentView==='timeline') renderTimeline(el);
  else if(currentView==='graph') renderGraph(el);
  else if(currentView==='hypotheses') renderHypotheses(el);
  else if(currentView==='audit') renderAudit(el);
}

function renderDashboard(el){
  el.innerHTML = `
  <div class="grid-4">
    <div class="stat-card blue"><div class="stat-label">Raw Events</div><div class="stat-value" style="color:var(--accent-blue)">2,847</div><div class="stat-detail">3 sources • immutable</div></div>
    <div class="stat-card violet"><div class="stat-label">Derived Events</div><div class="stat-value" style="color:var(--accent-violet)">1,203</div><div class="stat-detail">4 models • versioned</div></div>
    <div class="stat-card emerald"><div class="stat-label">Resolved Entities</div><div class="stat-value" style="color:var(--accent-emerald)">47</div><div class="stat-detail">12 persons • 8 vehicles</div></div>
    <div class="stat-card amber"><div class="stat-label">Active Hypotheses</div><div class="stat-value" style="color:var(--accent-amber)">3</div><div class="stat-detail">1 deterministic • 0 probabilistic</div></div>
  </div>
  <div class="grid-wide">
    <div class="panel">
      <div class="panel-header"><div class="panel-title">Recent Timeline</div><button class="btn" onclick="setView('timeline')">View All →</button></div>
      <div class="panel-body"><div class="timeline">${TIMELINE.slice(0,4).map(tlEntry).join('')}</div></div>
    </div>
    <div class="panel">
      <div class="panel-header"><div class="panel-title">System Health</div></div>
      <div class="panel-body">
        ${[
          {n:'API Server',v:'1.0.0',d:'< 200ms avg'},
          {n:'PostgreSQL + pgvector',v:'16',d:'10 conn pool'},
          {n:'Neo4j Community',v:'5.x',d:'Resolved only'},
          {n:'Redis Streams',v:'7.x',d:'Queue: 0'},
          {n:'Prometheus',v:'2.51',d:'/metrics'},
          {n:'Grafana',v:'10.4',d:':3000'},
        ].map(s=>`<div class="health-item"><div class="health-left"><div class="status-dot on"></div><div><div class="health-name">${s.n}</div><div class="health-detail">${s.d}</div></div></div><span class="tag model">${s.v}</span></div>`).join('')}
        <div class="model-info">
          <div class="model-info-title">Data Model</div>
          <div class="flow"><span class="tag raw">Layer 1: Raw</span><span class="flow-arrow">→</span><span class="tag derived">Layer 2: Derived</span><span class="flow-arrow">→</span><span class="tag inference">Layer 3: Inference</span></div>
          <div style="font-size:.6875rem;color:var(--text-tertiary);margin-top:.5rem;font-family:var(--font-mono)">Immutable raw events • Versioned outputs • Deterministic-first</div>
        </div>
      </div>
    </div>
  </div>`;
}

function tlEntry(e){
  return `<div class="tl-entry ${e.layer}">
    <div class="tl-meta">
      <span class="tl-time">${fmtTime(e.ts)}</span>
      <span class="tl-src">${e.src}</span>
      <span class="tag ${e.layer}">${e.layer.toUpperCase()}</span>
      <span class="tag ${confClass(e.conf)}">${(e.conf*100).toFixed(0)}%</span>
    </div>
    <div class="tl-summary">${e.sum}</div>
    <div class="tl-tags">
      <span class="tag model">${e.type}</span>
      ${e.mv?`<span class="tag model">⚙ ${e.mv}</span>`:''}
    </div>
    <div class="conf-bar"><div class="conf-fill ${confClass(e.conf)}" style="width:${e.conf*100}%"></div></div>
  </div>`;
}

function renderTimeline(el){
  const filtered = layerFilter ? TIMELINE.filter(e=>e.layer===layerFilter) : TIMELINE;
  el.innerHTML = `<div class="panel">
    <div class="panel-header">
      <div class="panel-title">Confidence-Weighted Timeline</div>
      <div class="filter-group">
        <button class="filter-pill ${!layerFilter?'active':''}" onclick="setFilter(null)">All</button>
        <button class="filter-pill ${layerFilter==='raw'?'active':''}" onclick="setFilter('raw')">Raw</button>
        <button class="filter-pill ${layerFilter==='derived'?'active':''}" onclick="setFilter('derived')">Derived</button>
        <button class="filter-pill ${layerFilter==='inference'?'active':''}" onclick="setFilter('inference')">Inference</button>
      </div>
    </div>
    <div class="panel-body"><div class="timeline">${filtered.map(tlEntry).join('')}</div></div>
  </div>`;
}

function renderGraph(el){
  const positions = calcPositions();
  let svg = `<svg viewBox="0 0 700 420">
    <defs><filter id="gl"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;

  EDGES.forEach(e=>{
    const from=positions[e.s], to=positions[e.t];
    if(!from||!to) return;
    const hl = selectedNode===e.s||selectedNode===e.t;
    svg += `<line x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}" stroke="${hl?'#06b6d4':'rgba(148,163,184,.15)'}" stroke-width="${hl?2:1}" ${e.conf<.7?'stroke-dasharray="4,4"':''}/>`;
    if(hl) svg += `<text x="${(from.x+to.x)/2}" y="${(from.y+to.y)/2-8}" fill="#94a3b8" font-size="8" text-anchor="middle" font-family="'JetBrains Mono',monospace">${e.rel} (${(e.conf*100).toFixed(0)}%)</text>`;
  });

  NODES.forEach(n=>{
    const p=positions[n.id]; if(!p) return;
    const c=NCOLORS[n.type]||'#64748b', sel=selectedNode===n.id, r=sel?20:14;
    if(sel) svg += `<circle cx="${p.x}" cy="${p.y}" r="${r+6}" fill="none" stroke="${c}" stroke-width="1" opacity=".3" filter="url(#gl)"/>`;
    svg += `<circle cx="${p.x}" cy="${p.y}" r="${r}" fill="${c}" fill-opacity="${selectedNode&&!sel?.3:.7}" stroke="${sel?'#fff':c}" stroke-width="${sel?2:1}" style="cursor:pointer" onclick="selectNode('${n.id}')"/>`;
    svg += `<text x="${p.x}" y="${p.y+r+12}" fill="${sel?'#f1f5f9':'#94a3b8'}" font-size="9" text-anchor="middle" font-family="'Inter',sans-serif" font-weight="${sel?600:400}">${n.label}</text>`;
    svg += `<text x="${p.x}" y="${p.y+3}" fill="#fff" font-size="7" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-weight="600">${n.type[0]}</text>`;
  });

  svg += '</svg>';

  let details = '';
  if(selectedNode){
    const node = NODES.find(n=>n.id===selectedNode);
    const edges = EDGES.filter(e=>e.s===selectedNode||e.t===selectedNode);
    if(node){
      details = `<div class="grid-2">
        <div class="panel"><div class="panel-header"><div class="panel-title">Node: ${node.label}</div></div><div class="panel-body">
          <div style="margin-bottom:.75rem"><span class="stat-label">Type</span><span class="tag ${node.type==='Person'?'raw':'derived'}">${node.type}</span></div>
          <div style="margin-bottom:.75rem"><span class="stat-label">Confidence</span><div style="font-size:1.375rem;font-weight:800;color:${confColor(node.conf)}">${(node.conf*100).toFixed(0)}%</div><div class="conf-bar"><div class="conf-fill ${confClass(node.conf)}" style="width:${node.conf*100}%"></div></div></div>
          ${node.mv?`<div style="margin-bottom:.75rem"><span class="stat-label">Model</span><span class="tag model">⚙ ${node.mv}</span></div>`:''}
          <div><span class="stat-label">Source Events</span><div style="font-family:var(--font-mono);font-size:.6875rem;color:var(--text-secondary)">${node.evts.length?node.evts.join(', '):'Infrastructure'}</div></div>
        </div></div>
        <div class="panel"><div class="panel-header"><div class="panel-title">Connected Edges</div></div><div class="panel-body">
          ${edges.map(e=>`<div class="hyp-card"><div style="display:flex;justify-content:space-between;margin-bottom:.5rem"><span class="tag inference">${e.rel}</span><span class="tag ${confClass(e.conf)}">${(e.conf*100).toFixed(0)}%</span></div><div style="font-size:.6875rem;color:var(--text-secondary);margin-bottom:4px"><b>Reason:</b> ${e.reason}</div><div style="font-size:.6875rem;color:var(--text-tertiary);font-family:var(--font-mono)">Model: ${e.model}</div></div>`).join('')}
        </div></div>
      </div>`;
    }
  }

  el.innerHTML = `<div class="panel">
    <div class="panel-header"><div class="panel-title">Entity Graph — Versioned Nodes & Edges</div><button class="btn" onclick="selectNode(null)">Reset</button></div>
    <div class="graph-box">${svg}</div>
    <div class="graph-legend">${Object.entries(NCOLORS).map(([t,c])=>`<div class="legend-item"><div class="legend-dot" style="background:${c}"></div>${t}</div>`).join('')}</div>
  </div>${details}`;
}

function calcPositions(){
  const cx=350,cy=210,pos={};
  const layout=[
    {ids:['p1'],r:70,a:Math.PI*1.5},{ids:['v1'],r:70,a:Math.PI*.5},
    {ids:['c1'],r:170,a:Math.PI*1.2},{ids:['c2'],r:170,a:Math.PI*1.8},{ids:['c3'],r:170,a:Math.PI*.3},
    {ids:['pl1'],r:160,a:Math.PI*.8},{ids:['ph1'],r:130,a:Math.PI*1.1}
  ];
  layout.forEach(g=>g.ids.forEach((id,i)=>{
    pos[id]={x:cx+Math.cos(g.a+i*.4)*g.r, y:cy+Math.sin(g.a+i*.4)*g.r};
  }));
  return pos;
}

function renderHypotheses(el){
  el.innerHTML = `<div class="grid-2">
    <div class="panel">
      <div class="panel-header"><div class="panel-title">Hypotheses — Deterministic First</div><button class="btn btn-primary">+ New</button></div>
      <div class="panel-body">
        <div class="hyp-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start"><div style="font-size:.8125rem;font-weight:600">${HYP.title}</div><span class="tag ${confClass(HYP.score)}">${(HYP.score*100).toFixed(0)}%</span></div>
          <div style="font-size:.6875rem;color:var(--text-secondary);line-height:1.5;margin:.5rem 0">${HYP.desc}</div>
          <div class="tl-tags" style="margin-bottom:.75rem"><span class="tag inference">Stage: ${HYP.stage}</span><span class="tag high">Status: ${HYP.status}</span></div>
          <div class="conf-bar" style="margin-bottom:.75rem"><div class="conf-fill ${confClass(HYP.score)}" style="width:${HYP.score*100}%"></div></div>
          <div class="scores-grid">${Object.entries(HYP.scores).map(([k,v])=>`<div class="score-item"><span class="score-label">${k}</span><span class="score-val" style="color:${confColor(v)}">${(v*100).toFixed(0)}%</span></div>`).join('')}</div>
        </div>
        <div class="disclaimer">⚠ Hypotheses are ranked suggestions with provenance — never automated accusations</div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-header"><div class="panel-title">Reasoning Pipeline</div></div>
      <div class="panel-body">
        ${[
          {s:'Stage 1: Deterministic',d:'Speed check • Time window • Plate exact • BBox overlap',st:'active',c:'var(--accent-emerald)'},
          {s:'Stage 2: Probabilistic',d:'Bayesian update • Embedding similarity • Sensor conf',st:'pending',c:'var(--accent-amber)'},
          {s:'Stage 3: Graph',d:'Multi-hop path • Structural consistency • Hypothesis scoring',st:'pending',c:'var(--accent-violet)'},
        ].map(s=>`<div class="pipeline-stage ${s.st==='active'?'active':''}" style="border-left-color:${s.c}"><div class="stage-title">${s.s}</div><div class="stage-desc">${s.d}</div><span class="tag ${s.st==='active'?'high':'model'}" style="margin-top:.5rem">${s.st}</span></div>`).join('')}
      </div>
    </div>
  </div>`;
}

function renderAudit(el){
  el.innerHTML = `<div class="panel">
    <div class="panel-header"><div class="panel-title">Audit Log — Append-Only, Immutable</div><button class="btn">Export</button></div>
    <div class="panel-body">
      <table class="audit-table"><thead><tr><th>ID</th><th>Time</th><th>Action</th><th>Actor</th><th>Resource</th><th>Model</th><th>Details</th></tr></thead><tbody>
      ${AUDIT.map(e=>`<tr><td>${e.id}</td><td>${e.ts}</td><td><span class="tag ${e.action==='inference'?'derived':'raw'}">${e.action}</span></td><td>${e.actor}</td><td>${e.res}</td><td>${e.mv?`<span class="tag model">${e.mv}</span>`:'—'}</td><td>${e.detail}</td></tr>`).join('')}
      </tbody></table>
      <div class="audit-footer"><span>Total: ${AUDIT.length}</span><span>UPDATE/DELETE blocked at DB</span><span>HMAC provenance: on</span></div>
    </div>
  </div>`;
}

// ── ACTIONS ──
function setView(v){currentView=v;render();}
function setFilter(f){layerFilter=f;render();}
function selectNode(id){selectedNode=selectedNode===id?null:id;render();}

// ── LIVE STATUS CHECK ──
async function checkStatus(){
  try{
    const r=await fetch('/health');
    if(r.ok){document.querySelectorAll('.status-dot').forEach(d=>d.className='status-dot on');}
  }catch(e){document.querySelectorAll('.status-dot').forEach(d=>d.className='status-dot off');}
}

// ── INIT ──
render();
setInterval(checkStatus,10000);
checkStatus();
</script>
</body>
</html>"""
