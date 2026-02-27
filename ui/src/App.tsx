import { useState } from 'react'

// ─── Types ──────────────────────────────────────────────────

interface TimelineEntry {
    id: string
    layer: 'raw' | 'derived' | 'inference'
    type: string
    timestamp: string
    source: string
    confidence: number
    summary: string
    modelVersion: string | null
    metadata: Record<string, unknown>
}

interface GraphNode {
    id: string
    label: string
    type: string
    modelVersion: string | null
    confidence: number
    sourceEventIds: string[]
}

interface GraphEdge {
    source: string
    target: string
    relationship: string
    confidence: number
    reason: string
    createdByModel: string
}

interface Hypothesis {
    id: string
    title: string
    description: string
    confidenceScore: number
    status: string
    reasoningStage: string
    scoringBreakdown: Record<string, number>
}

interface AuditLog {
    id: number
    action: string
    actor: string
    resourceType: string
    modelVersion: string | null
    details: string
    timestamp: string
}

// ─── Demo Data ──────────────────────────────────────────────

const DEMO_TIMELINE: TimelineEntry[] = [
    {
        id: '1', layer: 'raw', type: 'frame',
        timestamp: '2026-02-26T10:15:00Z', source: 'cam-001',
        confidence: 1.0,
        summary: 'Raw frame captured at Main St & 5th Ave — 1920×1080, SHA256 verified',
        modelVersion: null,
        metadata: { resolution: '1920x1080' },
    },
    {
        id: '2', layer: 'derived', type: 'detection',
        timestamp: '2026-02-26T10:15:00Z', source: 'cam-001',
        confidence: 0.92,
        summary: 'Person detected (92% conf) + white sedan (88% conf) — 3 detections total',
        modelVersion: 'yolov8n-8.3.0',
        metadata: { detections: 3 },
    },
    {
        id: '3', layer: 'derived', type: 'plate_read',
        timestamp: '2026-02-26T10:17:30Z', source: 'cam-001',
        confidence: 0.87,
        summary: 'License plate ABC-1234 recognized on white sedan — edit distance 0',
        modelVersion: 'paddleocr-2.8.0',
        metadata: { plate: 'ABC-1234' },
    },
    {
        id: '4', layer: 'derived', type: 'face_embedding',
        timestamp: '2026-02-26T10:22:00Z', source: 'cam-002',
        confidence: 0.78,
        summary: 'Face match (cosine sim 0.78) with subject from cam-001 — above threshold',
        modelVersion: 'insightface-buffalo_l-0.7.3',
        metadata: { cosine_similarity: 0.78 },
    },
    {
        id: '5', layer: 'raw', type: 'phone_ping',
        timestamp: '2026-02-26T10:25:00Z', source: 'cell-tower-42',
        confidence: 1.0,
        summary: 'Phone ping detected — 150m accuracy radius, signal strength -72 dBm',
        modelVersion: null,
        metadata: { accuracy_m: 150 },
    },
    {
        id: '6', layer: 'derived', type: 'plate_read',
        timestamp: '2026-02-26T10:45:00Z', source: 'cam-003',
        confidence: 0.94,
        summary: 'License plate ABC-1234 on Highway I-95 Mile 42 — exact match confirmed',
        modelVersion: 'paddleocr-2.8.0',
        metadata: { plate: 'ABC-1234', speed_mph: 65 },
    },
    {
        id: '7', layer: 'inference', type: 'constraint_check',
        timestamp: '2026-02-26T10:50:00Z', source: 'reasoner-v1',
        confidence: 0.85,
        summary: 'Speed check PASSED: 28 min travel, ~18 mi = ~39 mph (within 80 mph limit)',
        modelVersion: 'reasoner-deterministic-1.0',
        metadata: { stage: 'deterministic', result: 'feasible' },
    },
    {
        id: '8', layer: 'derived', type: 'face_embedding',
        timestamp: '2026-02-26T11:05:00Z', source: 'cam-002',
        confidence: 0.45,
        summary: 'Possible face match (0.45) — BELOW threshold, flagged for analyst review',
        modelVersion: 'insightface-buffalo_l-0.7.3',
        metadata: { below_threshold: true, occlusion: true },
    },
]

const DEMO_GRAPH_NODES: GraphNode[] = [
    { id: 'person-001', label: 'Unknown Subject A', type: 'Person', modelVersion: 'insightface-0.7.3', confidence: 0.72, sourceEventIds: ['de-001', 'de-003'] },
    { id: 'vehicle-001', label: 'White Sedan (ABC-1234)', type: 'Vehicle', modelVersion: 'paddleocr-2.8.0', confidence: 0.91, sourceEventIds: ['de-002', 'de-005'] },
    { id: 'cam-001', label: 'Main St & 5th', type: 'Camera', modelVersion: null, confidence: 1.0, sourceEventIds: [] },
    { id: 'cam-002', label: 'Park Entrance N', type: 'Camera', modelVersion: null, confidence: 1.0, sourceEventIds: [] },
    { id: 'cam-003', label: 'Hwy I-95 Mi42', type: 'Camera', modelVersion: null, confidence: 1.0, sourceEventIds: [] },
    { id: 'place-001', label: 'CBD', type: 'Place', modelVersion: null, confidence: 1.0, sourceEventIds: [] },
    { id: 'place-002', label: 'City Park', type: 'Place', modelVersion: null, confidence: 1.0, sourceEventIds: [] },
    { id: 'phone-001', label: 'Phone Ping', type: 'Phone', modelVersion: null, confidence: 0.65, sourceEventIds: ['re-004'] },
]

const DEMO_GRAPH_EDGES: GraphEdge[] = [
    { source: 'person-001', target: 'cam-001', relationship: 'APPEARED_IN', confidence: 0.92, reason: 'YOLOv8 + InsightFace', createdByModel: 'insightface-0.7.3' },
    { source: 'person-001', target: 'cam-002', relationship: 'APPEARED_IN', confidence: 0.78, reason: 'Face cosine sim 0.78', createdByModel: 'insightface-0.7.3' },
    { source: 'vehicle-001', target: 'cam-001', relationship: 'APPEARED_IN', confidence: 0.87, reason: 'PaddleOCR plate read', createdByModel: 'paddleocr-2.8.0' },
    { source: 'vehicle-001', target: 'cam-003', relationship: 'APPEARED_IN', confidence: 0.94, reason: 'Plate exact match', createdByModel: 'paddleocr-2.8.0' },
    { source: 'person-001', target: 'vehicle-001', relationship: 'NEAR', confidence: 0.75, reason: 'Co-located within 150s', createdByModel: 'reasoner-1.0' },
    { source: 'phone-001', target: 'person-001', relationship: 'NEAR', confidence: 0.55, reason: 'Phone ping 150m, 3min overlap', createdByModel: 'reasoner-1.0' },
    { source: 'cam-001', target: 'place-001', relationship: 'LOCATED_AT', confidence: 1.0, reason: 'Manual placement', createdByModel: 'manual' },
    { source: 'cam-002', target: 'place-002', relationship: 'LOCATED_AT', confidence: 1.0, reason: 'Manual placement', createdByModel: 'manual' },
]

const DEMO_HYPOTHESIS: Hypothesis = {
    id: 'hyp-001',
    title: 'Subject A traveled from CBD to Highway via Park',
    description: 'Unknown Subject A observed at Main St (10:15), Park Entrance (10:22), associated vehicle ABC-1234 detected on I-95 (10:45). Travel time and distance consistent.',
    confidenceScore: 0.73,
    status: 'active',
    reasoningStage: 'deterministic',
    scoringBreakdown: {
        'Temporal Consistency': 0.85,
        'Spatial Feasibility': 0.90,
        'Face Match Avg': 0.72,
        'LPR Match': 0.91,
        'Phone Corroboration': 0.55,
        'Combined Score': 0.73,
    },
}

const DEMO_AUDIT: AuditLog[] = [
    { id: 1, action: 'create', actor: 'system', resourceType: 'camera', modelVersion: null, details: '3 cameras initialized', timestamp: '09:00:00' },
    { id: 2, action: 'inference', actor: 'detector-v1', resourceType: 'detection', modelVersion: 'yolov8n-8.3.0', details: '45 frames → 12 detections', timestamp: '10:15:00' },
    { id: 3, action: 'inference', actor: 'face-embedder', resourceType: 'face_match', modelVersion: 'insightface-0.7.3', details: '2 matches, avg conf 0.78', timestamp: '10:20:00' },
    { id: 4, action: 'inference', actor: 'lpr-engine', resourceType: 'plate_read', modelVersion: 'paddleocr-2.8.0', details: 'ABC-1234 conf 0.91', timestamp: '10:17:30' },
    { id: 5, action: 'create', actor: 'analyst-demo', resourceType: 'hypothesis', modelVersion: null, details: 'Subject A travel hypothesis', timestamp: '11:30:00' },
    { id: 6, action: 'inference', actor: 'reasoner-v1', resourceType: 'constraint_check', modelVersion: 'deterministic-1.0', details: 'Speed check passed', timestamp: '10:50:00' },
]

// ─── Helpers ────────────────────────────────────────────────

function getConfidenceClass(conf: number): string {
    if (conf >= 0.75) return 'high'
    if (conf >= 0.5) return 'medium'
    return 'low'
}

function formatTime(ts: string): string {
    return new Date(ts).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

// ─── SVG Graph Renderer ─────────────────────────────────────

const NODE_COLORS: Record<string, string> = {
    Person: '#3b82f6',
    Vehicle: '#f59e0b',
    Camera: '#10b981',
    Place: '#8b5cf6',
    Phone: '#06b6d4',
}

interface NodePosition { x: number; y: number }

function getNodePositions(nodes: GraphNode[]): Map<string, NodePosition> {
    const positions = new Map<string, NodePosition>()
    const centerX = 400
    const centerY = 250

    // Layout: circular with type-based grouping
    const groups: Record<string, GraphNode[]> = {}
    nodes.forEach(n => {
        if (!groups[n.type]) groups[n.type] = []
        groups[n.type].push(n)
    })

    const typeKeys = Object.keys(groups)
    typeKeys.forEach((type, ti) => {
        const group = groups[type]
        const angleOffset = (ti / typeKeys.length) * Math.PI * 2
        const radius = type === 'Camera' || type === 'Place' ? 180 : 100

        group.forEach((node, ni) => {
            const angle = angleOffset + (ni / Math.max(group.length, 1)) * (Math.PI * 2 / typeKeys.length)
            positions.set(node.id, {
                x: centerX + Math.cos(angle) * radius,
                y: centerY + Math.sin(angle) * radius,
            })
        })
    })

    return positions
}

function GraphVisualization({ nodes, edges, selectedNode, onSelectNode }: {
    nodes: GraphNode[]
    edges: GraphEdge[]
    selectedNode: string | null
    onSelectNode: (id: string | null) => void
}) {
    const positions = getNodePositions(nodes)

    return (
        <div className="graph-container">
            <svg className="graph-canvas" viewBox="0 0 800 500">
                <defs>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                    {Object.entries(NODE_COLORS).map(([type, color]) => (
                        <radialGradient key={type} id={`grad-${type}`}>
                            <stop offset="0%" stopColor={color} stopOpacity="0.9" />
                            <stop offset="100%" stopColor={color} stopOpacity="0.5" />
                        </radialGradient>
                    ))}
                </defs>

                {/* Edges */}
                {edges.map((edge, i) => {
                    const from = positions.get(edge.source)
                    const to = positions.get(edge.target)
                    if (!from || !to) return null
                    const isHighlighted = selectedNode === edge.source || selectedNode === edge.target
                    return (
                        <g key={i}>
                            <line
                                x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                                stroke={isHighlighted ? '#06b6d4' : 'rgba(148,163,184,0.15)'}
                                strokeWidth={isHighlighted ? 2 : 1}
                                strokeDasharray={edge.confidence < 0.7 ? '4,4' : 'none'}
                            />
                            {isHighlighted && (
                                <text
                                    x={(from.x + to.x) / 2}
                                    y={(from.y + to.y) / 2 - 8}
                                    fill="#94a3b8"
                                    fontSize="9"
                                    textAnchor="middle"
                                    fontFamily="'JetBrains Mono', monospace"
                                >
                                    {edge.relationship} ({(edge.confidence * 100).toFixed(0)}%)
                                </text>
                            )}
                        </g>
                    )
                })}

                {/* Nodes */}
                {nodes.map(node => {
                    const pos = positions.get(node.id)
                    if (!pos) return null
                    const color = NODE_COLORS[node.type] || '#64748b'
                    const isSelected = selectedNode === node.id
                    const radius = isSelected ? 22 : 16

                    return (
                        <g key={node.id} onClick={() => onSelectNode(isSelected ? null : node.id)} style={{ cursor: 'pointer' }}>
                            {isSelected && (
                                <circle cx={pos.x} cy={pos.y} r={radius + 6} fill="none" stroke={color} strokeWidth="1" opacity="0.3" filter="url(#glow)" />
                            )}
                            <circle
                                cx={pos.x} cy={pos.y} r={radius}
                                fill={`url(#grad-${node.type})`}
                                stroke={isSelected ? 'white' : color}
                                strokeWidth={isSelected ? 2 : 1}
                                opacity={selectedNode && !isSelected ? 0.4 : 1}
                            />
                            <text
                                x={pos.x} y={pos.y + radius + 14}
                                fill={isSelected ? '#f1f5f9' : '#94a3b8'}
                                fontSize="10"
                                textAnchor="middle"
                                fontFamily="'Inter', sans-serif"
                                fontWeight={isSelected ? 600 : 400}
                            >
                                {node.label}
                            </text>
                            <text
                                x={pos.x} y={pos.y + 4}
                                fill="white"
                                fontSize="8"
                                textAnchor="middle"
                                fontFamily="'JetBrains Mono', monospace"
                                fontWeight={600}
                            >
                                {node.type.charAt(0)}
                            </text>
                        </g>
                    )
                })}
            </svg>

            <div className="graph-legend">
                {Object.entries(NODE_COLORS).map(([type, color]) => (
                    <div key={type} className="legend-item">
                        <div className="legend-dot" style={{ background: color }} />
                        {type}
                    </div>
                ))}
            </div>
        </div>
    )
}

// ─── Main App ───────────────────────────────────────────────

type View = 'dashboard' | 'timeline' | 'graph' | 'hypotheses' | 'audit'

export default function App() {
    const [view, setView] = useState<View>('dashboard')
    const [layerFilter, setLayerFilter] = useState<string | null>(null)
    const [selectedNode, setSelectedNode] = useState<string | null>(null)

    const filteredTimeline = layerFilter
        ? DEMO_TIMELINE.filter(e => e.layer === layerFilter)
        : DEMO_TIMELINE

    return (
        <div className="app">
            {/* ── Header ─────────────────────────────────────────── */}
            <header className="header">
                <div className="header-brand">
                    <div className="header-logo">S</div>
                    <div>
                        <div className="header-title">Sentinel</div>
                        <div className="header-subtitle">Spatiotemporal Entity Reasoning Engine</div>
                    </div>
                </div>

                <nav className="header-nav">
                    {(['dashboard', 'timeline', 'graph', 'hypotheses', 'audit'] as View[]).map(v => (
                        <button
                            key={v}
                            className={`nav-btn ${view === v ? 'active' : ''}`}
                            onClick={() => setView(v)}
                        >
                            {v.charAt(0).toUpperCase() + v.slice(1)}
                        </button>
                    ))}
                </nav>

                <div className="header-status">
                    <div className="status-indicator">
                        <div className="status-dot online" />
                        API
                    </div>
                    <div className="status-indicator">
                        <div className="status-dot online" />
                        PostGIS
                    </div>
                    <div className="status-indicator">
                        <div className="status-dot online" />
                        Neo4j
                    </div>
                    <div className="status-indicator">
                        <div className="status-dot online" />
                        Redis
                    </div>
                </div>
            </header>

            {/* ── Main Content ───────────────────────────────────── */}
            <main className="main-content">
                {/* ── DASHBOARD VIEW ─────────────────────────────── */}
                {view === 'dashboard' && (
                    <>
                        <div className="dashboard-grid">
                            <div className="stat-card blue">
                                <div className="stat-label">Raw Events</div>
                                <div className="stat-value" style={{ color: 'var(--accent-blue)' }}>2,847</div>
                                <div className="stat-detail">3 sources • immutable</div>
                            </div>
                            <div className="stat-card violet">
                                <div className="stat-label">Derived Events</div>
                                <div className="stat-value" style={{ color: 'var(--accent-violet)' }}>1,203</div>
                                <div className="stat-detail">4 models • versioned</div>
                            </div>
                            <div className="stat-card emerald">
                                <div className="stat-label">Resolved Entities</div>
                                <div className="stat-value" style={{ color: 'var(--accent-emerald)' }}>47</div>
                                <div className="stat-detail">12 persons • 8 vehicles</div>
                            </div>
                            <div className="stat-card amber">
                                <div className="stat-label">Active Hypotheses</div>
                                <div className="stat-value" style={{ color: 'var(--accent-amber)' }}>3</div>
                                <div className="stat-detail">1 deterministic • 0 probabilistic</div>
                            </div>
                        </div>

                        <div className="content-wide">
                            {/* Timeline Preview */}
                            <div className="panel">
                                <div className="panel-header">
                                    <div className="panel-title">Recent Timeline</div>
                                    <button className="btn btn-ghost" onClick={() => setView('timeline')}>View All →</button>
                                </div>
                                <div className="panel-body">
                                    <div className="timeline">
                                        {DEMO_TIMELINE.slice(0, 4).map(entry => (
                                            <div key={entry.id} className={`timeline-entry layer-${entry.layer}`}>
                                                <div className="timeline-meta">
                                                    <span className="timeline-time">{formatTime(entry.timestamp)}</span>
                                                    <span className="timeline-source">{entry.source}</span>
                                                    <span className={`tag tag-layer ${entry.layer}`}>{entry.layer}</span>
                                                    <span className={`tag tag-confidence ${getConfidenceClass(entry.confidence)}`}>
                                                        {(entry.confidence * 100).toFixed(0)}%
                                                    </span>
                                                </div>
                                                <div className="timeline-summary">{entry.summary}</div>
                                                {entry.modelVersion && (
                                                    <div className="timeline-tags">
                                                        <span className="tag tag-model">⚙ {entry.modelVersion}</span>
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            {/* System Health */}
                            <div className="panel">
                                <div className="panel-header">
                                    <div className="panel-title">System Health</div>
                                </div>
                                <div className="panel-body">
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
                                        {[
                                            { name: 'API Server', status: 'up', version: '1.0.0', detail: '< 200ms avg' },
                                            { name: 'PostgreSQL + pgvector', status: 'up', version: '16 + pgvector', detail: '10 conn pool' },
                                            { name: 'Neo4j Community', status: 'up', version: '5.x', detail: 'Resolved entities only' },
                                            { name: 'Redis Streams', status: 'up', version: '7.x', detail: 'Queue depth: 0' },
                                            { name: 'Prometheus', status: 'up', version: '2.51', detail: 'Scraping /metrics' },
                                            { name: 'Grafana', status: 'up', version: '10.4', detail: ':3000' },
                                        ].map(s => (
                                            <div key={s.name} style={{
                                                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                                padding: 'var(--space-3)', background: 'var(--bg-tertiary)',
                                                borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)',
                                            }}>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                                                    <div className={`status-dot ${s.status === 'up' ? 'online' : 'offline'}`} />
                                                    <div>
                                                        <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600 }}>{s.name}</div>
                                                        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
                                                            {s.detail}
                                                        </div>
                                                    </div>
                                                </div>
                                                <span className="tag tag-model">{s.version}</span>
                                            </div>
                                        ))}
                                    </div>

                                    {/* Data Model Info */}
                                    <div style={{
                                        marginTop: 'var(--space-4)', padding: 'var(--space-3)',
                                        background: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)',
                                        border: '1px solid var(--border-subtle)',
                                    }}>
                                        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)', marginBottom: 'var(--space-2)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                                            Data Model
                                        </div>
                                        <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                                            <span className="tag tag-layer raw">Layer 1: Raw</span>
                                            <span style={{ color: 'var(--text-muted)' }}>→</span>
                                            <span className="tag tag-layer derived">Layer 2: Derived</span>
                                            <span style={{ color: 'var(--text-muted)' }}>→</span>
                                            <span className="tag tag-layer inference">Layer 3: Inference</span>
                                        </div>
                                        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)', marginTop: 'var(--space-2)', fontFamily: 'var(--font-mono)' }}>
                                            Immutable raw events • Versioned model outputs • Deterministic-first reasoning
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </>
                )}

                {/* ── TIMELINE VIEW ──────────────────────────────── */}
                {view === 'timeline' && (
                    <div className="panel">
                        <div className="panel-header">
                            <div className="panel-title">Confidence-Weighted Timeline</div>
                            <div className="panel-controls">
                                <div className="filter-group">
                                    <button className={`filter-pill ${!layerFilter ? 'active' : ''}`} onClick={() => setLayerFilter(null)}>All</button>
                                    <button className={`filter-pill ${layerFilter === 'raw' ? 'active' : ''}`} onClick={() => setLayerFilter('raw')}>Raw</button>
                                    <button className={`filter-pill ${layerFilter === 'derived' ? 'active' : ''}`} onClick={() => setLayerFilter('derived')}>Derived</button>
                                    <button className={`filter-pill ${layerFilter === 'inference' ? 'active' : ''}`} onClick={() => setLayerFilter('inference')}>Inference</button>
                                </div>
                            </div>
                        </div>
                        <div className="panel-body">
                            <div className="timeline">
                                {filteredTimeline.map(entry => (
                                    <div key={entry.id} className={`timeline-entry layer-${entry.layer}`}>
                                        <div className="timeline-meta">
                                            <span className="timeline-time">{formatTime(entry.timestamp)}</span>
                                            <span className="timeline-source">{entry.source}</span>
                                            <span className={`tag tag-layer ${entry.layer}`}>{entry.layer.toUpperCase()}</span>
                                            <span className={`tag tag-confidence ${getConfidenceClass(entry.confidence)}`}>
                                                {(entry.confidence * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                        <div className="timeline-summary">{entry.summary}</div>
                                        <div className="timeline-tags">
                                            <span className="tag" style={{ background: 'rgba(148,163,184,0.06)', color: 'var(--text-tertiary)' }}>
                                                {entry.type}
                                            </span>
                                            {entry.modelVersion && (
                                                <span className="tag tag-model">⚙ {entry.modelVersion}</span>
                                            )}
                                        </div>
                                        <div className="confidence-bar">
                                            <div
                                                className={`confidence-fill ${getConfidenceClass(entry.confidence)}`}
                                                style={{ width: `${entry.confidence * 100}%` }}
                                            />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                {/* ── GRAPH VIEW ─────────────────────────────────── */}
                {view === 'graph' && (
                    <>
                        <div className="panel">
                            <div className="panel-header">
                                <div className="panel-title">Entity Graph — Versioned Nodes & Edges</div>
                                <div className="panel-controls">
                                    <button className="btn" onClick={() => setSelectedNode(null)}>Reset Selection</button>
                                </div>
                            </div>
                            <GraphVisualization
                                nodes={DEMO_GRAPH_NODES}
                                edges={DEMO_GRAPH_EDGES}
                                selectedNode={selectedNode}
                                onSelectNode={setSelectedNode}
                            />
                        </div>

                        {/* Node/Edge Details */}
                        {selectedNode && (
                            <div className="content-split">
                                <div className="panel">
                                    <div className="panel-header">
                                        <div className="panel-title">Node: {DEMO_GRAPH_NODES.find(n => n.id === selectedNode)?.label}</div>
                                    </div>
                                    <div className="panel-body">
                                        {(() => {
                                            const node = DEMO_GRAPH_NODES.find(n => n.id === selectedNode)
                                            if (!node) return null
                                            return (
                                                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
                                                    <div>
                                                        <span className="stat-label">Type</span>
                                                        <div className={`tag tag-layer ${node.type === 'Person' ? 'raw' : node.type === 'Vehicle' ? 'derived' : 'inference'}`}>{node.type}</div>
                                                    </div>
                                                    <div>
                                                        <span className="stat-label">Confidence</span>
                                                        <div className="stat-value" style={{ fontSize: 'var(--text-xl)' }}>{(node.confidence * 100).toFixed(0)}%</div>
                                                        <div className="confidence-bar">
                                                            <div className={`confidence-fill ${getConfidenceClass(node.confidence)}`} style={{ width: `${node.confidence * 100}%` }} />
                                                        </div>
                                                    </div>
                                                    {node.modelVersion && (
                                                        <div>
                                                            <span className="stat-label">Model Version</span>
                                                            <span className="tag tag-model">⚙ {node.modelVersion}</span>
                                                        </div>
                                                    )}
                                                    <div>
                                                        <span className="stat-label">Source Events</span>
                                                        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>
                                                            {node.sourceEventIds.length > 0 ? node.sourceEventIds.join(', ') : 'Infrastructure node'}
                                                        </div>
                                                    </div>
                                                </div>
                                            )
                                        })()}
                                    </div>
                                </div>
                                <div className="panel">
                                    <div className="panel-header">
                                        <div className="panel-title">Connected Edges</div>
                                    </div>
                                    <div className="panel-body">
                                        {DEMO_GRAPH_EDGES
                                            .filter(e => e.source === selectedNode || e.target === selectedNode)
                                            .map((edge, i) => (
                                                <div key={i} className="hypothesis-card">
                                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-2)' }}>
                                                        <span className="tag" style={{ background: 'var(--accent-cyan-glow)', color: 'var(--accent-cyan)' }}>{edge.relationship}</span>
                                                        <span className={`tag tag-confidence ${getConfidenceClass(edge.confidence)}`}>{(edge.confidence * 100).toFixed(0)}%</span>
                                                    </div>
                                                    <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)', marginBottom: 'var(--space-1)' }}>
                                                        <strong>Reason:</strong> {edge.reason}
                                                    </div>
                                                    <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
                                                        Model: {edge.createdByModel}
                                                    </div>
                                                </div>
                                            ))
                                        }
                                    </div>
                                </div>
                            </div>
                        )}
                    </>
                )}

                {/* ── HYPOTHESES VIEW ────────────────────────────── */}
                {view === 'hypotheses' && (
                    <div className="content-split">
                        <div className="panel">
                            <div className="panel-header">
                                <div className="panel-title">Hypotheses — Deterministic First</div>
                                <button className="btn btn-primary">+ New Hypothesis</button>
                            </div>
                            <div className="panel-body">
                                <div className="hypothesis-card">
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                        <div className="hypothesis-title">{DEMO_HYPOTHESIS.title}</div>
                                        <span className={`tag tag-confidence ${getConfidenceClass(DEMO_HYPOTHESIS.confidenceScore)}`}>
                                            {(DEMO_HYPOTHESIS.confidenceScore * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                    <div className="hypothesis-description">{DEMO_HYPOTHESIS.description}</div>
                                    <div className="timeline-tags" style={{ marginBottom: 'var(--space-3)' }}>
                                        <span className="tag tag-layer inference">Stage: {DEMO_HYPOTHESIS.reasoningStage}</span>
                                        <span className="tag" style={{ background: 'var(--accent-emerald-glow)', color: 'var(--accent-emerald)' }}>
                                            Status: {DEMO_HYPOTHESIS.status}
                                        </span>
                                    </div>
                                    <div className="confidence-bar" style={{ marginBottom: 'var(--space-3)' }}>
                                        <div
                                            className={`confidence-fill ${getConfidenceClass(DEMO_HYPOTHESIS.confidenceScore)}`}
                                            style={{ width: `${DEMO_HYPOTHESIS.confidenceScore * 100}%` }}
                                        />
                                    </div>
                                    <div className="hypothesis-scores" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
                                        {Object.entries(DEMO_HYPOTHESIS.scoringBreakdown).map(([key, value]) => (
                                            <div key={key} className="score-item">
                                                <span className="score-label">{key}</span>
                                                <span className={`score-value`} style={{ color: `var(--conf-${getConfidenceClass(value)})` }}>
                                                    {(value * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div style={{
                                    padding: 'var(--space-3)', background: 'var(--bg-primary)',
                                    borderRadius: 'var(--radius-sm)', border: '1px dashed var(--border-default)',
                                    textAlign: 'center', color: 'var(--text-tertiary)', fontSize: 'var(--text-xs)',
                                }}>
                                    ⚠ Hypotheses are ranked suggestions with provenance — never automated accusations
                                </div>
                            </div>
                        </div>

                        <div className="panel">
                            <div className="panel-header">
                                <div className="panel-title">Reasoning Pipeline</div>
                            </div>
                            <div className="panel-body">
                                {[
                                    { stage: 'Stage 1: Deterministic', desc: 'Speed violation • Time window • Plate exact match • BBox overlap', status: 'active', color: 'var(--accent-emerald)' },
                                    { stage: 'Stage 2: Probabilistic', desc: 'Bayesian belief update • Embedding similarity • Sensor confidence', status: 'pending', color: 'var(--accent-amber)' },
                                    { stage: 'Stage 3: Graph', desc: 'Multi-hop path search • Structural consistency • Hypothesis scoring', status: 'pending', color: 'var(--accent-violet)' },
                                ].map((s, i) => (
                                    <div key={i} style={{
                                        padding: 'var(--space-4)', marginBottom: 'var(--space-3)',
                                        background: s.status === 'active' ? 'rgba(16, 185, 129, 0.05)' : 'var(--bg-tertiary)',
                                        border: `1px solid ${s.status === 'active' ? 'rgba(16, 185, 129, 0.2)' : 'var(--border-subtle)'}`,
                                        borderRadius: 'var(--radius-md)',
                                        borderLeft: `3px solid ${s.color}`,
                                    }}>
                                        <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, marginBottom: 'var(--space-1)' }}>{s.stage}</div>
                                        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>{s.desc}</div>
                                        <span className="tag" style={{
                                            marginTop: 'var(--space-2)',
                                            background: s.status === 'active' ? 'var(--accent-emerald-glow)' : 'var(--bg-primary)',
                                            color: s.status === 'active' ? 'var(--accent-emerald)' : 'var(--text-muted)',
                                        }}>
                                            {s.status}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                {/* ── AUDIT VIEW ─────────────────────────────────── */}
                {view === 'audit' && (
                    <div className="panel">
                        <div className="panel-header">
                            <div className="panel-title">Audit Log — Append-Only, Immutable</div>
                            <div className="panel-controls">
                                <button className="btn">Export Log</button>
                            </div>
                        </div>
                        <div className="panel-body">
                            <table className="audit-table">
                                <thead>
                                    <tr>
                                        <th>ID</th>
                                        <th>Timestamp</th>
                                        <th>Action</th>
                                        <th>Actor</th>
                                        <th>Resource</th>
                                        <th>Model Version</th>
                                        <th>Details</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {DEMO_AUDIT.map(entry => (
                                        <tr key={entry.id}>
                                            <td>{entry.id}</td>
                                            <td>{entry.timestamp}</td>
                                            <td>
                                                <span className="tag" style={{
                                                    background: entry.action === 'inference' ? 'var(--accent-violet-glow)' : 'var(--accent-blue-glow)',
                                                    color: entry.action === 'inference' ? 'var(--accent-violet)' : 'var(--accent-blue)',
                                                }}>
                                                    {entry.action}
                                                </span>
                                            </td>
                                            <td>{entry.actor}</td>
                                            <td>{entry.resourceType}</td>
                                            <td>{entry.modelVersion ? <span className="tag tag-model">{entry.modelVersion}</span> : '—'}</td>
                                            <td style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis' }}>{entry.details}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>

                            <div style={{
                                marginTop: 'var(--space-4)', padding: 'var(--space-3)',
                                background: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)',
                                border: '1px solid var(--border-subtle)',
                                display: 'flex', justifyContent: 'space-between',
                                fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)',
                            }}>
                                <span>Total entries: {DEMO_AUDIT.length}</span>
                                <span>UPDATE/DELETE triggers blocked at DB level</span>
                                <span>HMAC provenance signing: enabled</span>
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    )
}
