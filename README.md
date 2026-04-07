---
title: Wellbeing Feed RL Environment
emoji: 🧠
colorFrom: teal
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# Wellbeing Feed RL Environment

An RL environment simulating social media content recommendation that balances user engagement with mental wellbeing.

## Problem

Social media algorithms optimize for engagement — but the most engaging content is often toxic, creating doomscroll spirals that damage mental health.

## Observation Space (8 signals + 10 history = 17D vector)

| Signal              | Range     | Description             |
| ------------------- | --------- | ----------------------- |
| user_mood           | [-1, 1]   | Emotional state         |
| mood_trend          | [-1, 1]   | Rate of mood change     |
| engagement_level    | [0, 1]    | User engagement         |
| content_history     | 10x[-1,1] | Last 10 item sentiments |
| session_duration    | [0, 1]    | Session progress        |
| vulnerability_score | [0, 1]    | Spiral susceptibility   |
| scroll_velocity     | [0, 1]    | Scroll speed            |
| time_of_day         | [0, 1]    | Time of day             |

## Action Space (5 discrete)

| ID  | Category     | Engagement | Mood          | Risk     |
| --- | ------------ | ---------- | ------------- | -------- |
| 0   | Trending     | High       | Negative      | High     |
| 1   | Motivational | Medium     | Very Positive | Low      |
| 2   | Educational  | Low        | Mild Positive | Very Low |
| 3   | Funny        | Med-High   | Positive      | Low      |
| 4   | Personalized | High       | Mixed         | Medium   |

## Reward Function (0.0-1.0, 6 components)

30% mood health + 25% engagement + 15% mood improvement + 10% anti-doomscroll + spiral rescue bonus - toxicity penalty

## Tasks

| Task                    | Steps | Threshold | Challenge                            |
| ----------------------- | ----- | --------- | ------------------------------------ |
| easy_stable_user        | 50    | 0.6       | Maintain mood + engagement           |
| medium_spiral_detection | 100   | 0.5       | Detect and break negative spirals    |
| hard_engagement_trap    | 150   | 0.4       | Break doomscroll without losing user |

## Baseline Scores (seed=42)

| Agent           | Easy  | Medium | Hard  | Overall |
| --------------- | ----- | ------ | ----- | ------- |
| Heuristic       | 0.755 | 0.702  | 0.676 | 0.705   |
| Random          | 0.743 | 0.672  | 0.619 | 0.668   |
| Q-Learning      | 0.758 | 0.596  | 0.572 | 0.627   |
| Engagement-Only | 0.645 | 0.585  | 0.549 | 0.586   |

## Setup

```
pip install -r requirements.txt
python run_baseline.py --seed 42
python inference.py
```

## Environment Variables

- API_BASE_URL: LLM API endpoint
- MODEL_NAME: Model identifier
- HF_TOKEN: Hugging Face API key

## API Endpoints

- POST /reset - Reset environment
- POST /step - Take action
- GET /state - Current state
- GET /tasks - List tasks
