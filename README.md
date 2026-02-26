Chronexus is an open-source incident reconstruction engine that ingests multi-modal data (video, metadata, location signals, structured logs) and converts it into a unified, queryable entity graph.

The system is designed to assist human analysts in reconstructing events by:

Extracting entities from video and sensor streams

Linking entities across time and space

Scoring hypotheses using probabilistic reasoning

Providing transparent, auditable evidence trails

Chronexus does not automate accusations or decisions. It is a decision-support system designed to surface structured insights while preserving uncertainty and provenance.

Core Capabilities

Multi-camera video ingestion (RTSP / MP4)

Object detection and multi-object tracking

Face and vehicle embedding extraction

License plate recognition (LPR)

Geospatial normalization (WGS84)

Entity graph construction (Neo4j)

Temporal consistency validation

Bayesian confidence scoring

Interactive graph and timeline exploration

Immutable audit logs and evidence provenance tracking

Architecture Overview

Chronexus follows a modular pipeline:

Data Ingestion

Feature Extraction

Event Normalization

Graph Construction

Probabilistic Reasoning

Analyst Interface

Each module runs as an isolated service and communicates via event streams.

The system is designed to be:

Extensible

Vendor-neutral

Cloud-agnostic

Fully open-source compatible

Design Principles

Human-in-the-loop by default

Explicit uncertainty modeling

Transparent inference paths

Minimal assumptions

Reproducible pipelines

Data provenance preserved at every stage

Intended Use Cases

Incident reconstruction

Fraud investigation

Insurance claim analysis

Corporate security operations

Research in spatiotemporal reasoning systems

Not a Surveillance Platform

Chronexus is not designed for mass surveillance or automated criminal identification.
It is a structured reasoning engine intended for controlled, legally compliant environments.
