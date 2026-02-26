Chronexus
Multi-Modal Incident Reconstruction & Evidence Correlation Engine

Chronexus is an open-source, human-in-the-loop investigation support system designed to reconstruct incidents from fragmented, multi-modal data sources.

It ingests video streams, metadata, structured logs, and sensor data, transforms them into normalized events, and builds a spatiotemporal entity graph to assist analysts in reconstructing timelines and identifying relationships between people, vehicles, locations, and events.

Chronexus does not automate accusations.
It provides structured reasoning, confidence-weighted hypotheses, and transparent evidence linkage to support informed human decision-making.

Core Capabilities

Real-time video ingestion and frame processing

Multi-object detection and tracking (persons, vehicles, plates)

Face and vehicle embedding extraction

License plate recognition (OCR)

Unified event schema normalization

Graph-based entity linking (Neo4j)

Spatiotemporal consistency validation

Bayesian confidence scoring

Interactive timeline and graph exploration

Evidence provenance and audit logging

System Architecture

Chronexus is structured as modular services:

Ingestion Layer – Connectors for video streams, logs, and structured metadata

Feature Extraction Layer – Detection, tracking, embeddings, OCR

Event Normalization Engine – Converts raw outputs into a unified schema

Entity Graph Layer – Graph database for relationship modeling

Reasoning Engine – Hypothesis scoring and temporal validation

Analyst Interface – Timeline, graph explorer, and evidence viewer

All components are containerized and orchestrated using Docker.

Technology Stack

Python (FastAPI, async workers)

Neo4j (graph database)

PostgreSQL + PostGIS (geospatial data)

Redis Streams (event queue)

YOLOv8 / Detectron2 (object detection)

InsightFace / FaceNet (face embeddings)

PaddleOCR / OpenALPR (license plate recognition)

React + TypeScript (frontend)

Docker / Docker Compose

All dependencies are open-source and free to use.

Design Principles

Human-in-the-loop decision support

Transparency over automation

Confidence-weighted inference

Modular, extensible architecture

Privacy-aware and audit-traceable

Dataset-agnostic ingestion pipeline

Use Cases

Chronexus is designed as a general-purpose evidence correlation engine and can be applied to:

Insurance investigations

Corporate incident analysis

Campus or facility security

Fraud detection workflows

Operational anomaly detection

The architecture is domain-flexible and not restricted to law enforcement contexts.

Project Status

This project is under active development.
The current focus is building the ingestion pipeline, event normalization engine, and foundational entity graph.

Disclaimer

Chronexus is a research and infrastructure project intended for ethical, lawful, and privacy-compliant use only.
It does not provide automated determinations of guilt or liability. All outputs require human review and validation.
