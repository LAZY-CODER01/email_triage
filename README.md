---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# Email Triage Environment

## Description
Models a genuine Level 1 Customer Support pipeline. Agents must categorize technical vs billing issues, prioritize urgent downtime over routine requests, and draft empathetic, policy-compliant responses to angry users.

## Action & Observation Spaces
* **Observation**: `TriageObservation` containing current task difficulty, instructions, and an array of `Email` objects.
* **Action**: `TriageAction` mapping email IDs to categories/priorities, or providing a string draft response.