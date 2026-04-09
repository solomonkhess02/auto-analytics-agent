# 📘 Phase 2 — Complete Learning Reference

### Agentic Data Cleaner & Human-in-the-Loop Orchestration (`auto-analytics-agent`)

> **Purpose of this document:** A self-contained, textbook-style reference for everything built in Phase 2. This document explains the architectural upgrades that transformed our pipeline from a rigid automation script into a true "Agentic System" capable of pausing for human review and validating its own work.

---

## 📋 Table of Contents

1. [Phase 2 Vision — Automation vs. Agentic AI](#1-phase-2-vision--automation-vs-agentic-ai)
2. [The Architecture Shift — "Copilot Mode"](#2-the-architecture-shift--copilot-mode)
3. [Step 8: Centralizing the Brain (`core/prompts.py`)](#3-step-8-centralizing-the-brain-corepromptspy)
4. [Step 9: Agentic Reasoning & Planning (`agents/data_cleaner.py`)](#4-step-9-agentic-reasoning--planning-agentsdata_cleanerpy)
5. [Step 10: Human-in-the-Loop Orchestration (`core/graph.py`)](#5-step-10-human-in-the-loop-orchestration-coregraphpy)
6. [Step 11: Dynamic Execution & The Validation Loop](#6-step-11-dynamic-execution--the-validation-loop)
7. [Step 12: Integration Testing & Simulation](#7-step-12-integration-testing--simulation)
8. [Concept Deep Dive: LangGraph Checkpointing](#8-concept-deep-dive-langgraph-checkpointing)
9. [Bugs & Fixes Encountered During Phase 2](#9-bugs--fixes-encountered-during-phase-2)

---

## 1. Phase 2 Vision — Automation vs. Agentic AI

### The Danger of Blind Automation

In Phase 1, our Data Profiler was a single-shot generator: it looked at the dataset, generated Python code, executed it, and returned the result. This works well for _reading_ data (profiling is non-destructive).

However, in Phase 2, we introduced the **Data Cleaner Agent**. Cleaning data is destructive. If an LLM independently decides to drop 5 important columns or impute missing financial figures with zeros without human oversight, it ruins the entire analysis.

### Our Solution: The "Plan-Then-Act" Agent

We solved this by splitting the Cleaner Agent into two distinct cognitive steps:

1. **The Planner:** Analyzes the data and produces a structured JSON blueprint of what it _intends_ to do.
2. **The Executor:** Takes the approved instruction and writes the code to permanently manipulate the dataset.

This split enables true collaborative AI.

---

## 2. The Architecture Shift — "Copilot Mode"

To support human-in-the-loop validation, we modified our LangGraph execution pipeline.

```text
┌──────────────────────────────────────────────────────────────────┐
│                   Phase 2 Orchestration Flow                     │
│                                                                  │
│  [START] ──▶ [Data Profiler] ──▶ [Cleaner Planner]               │
│                                           │                      │
│                                           ▼                      │
│                                     [ INTERRUPT ]  ◀── (Human    │
│                                           │            Feedback) │
│                                           ▼                      │
│                                     [Cleaner Executor] ──▶ [END] │
└──────────────────────────────────────────────────────────────────┘
```
