"""Agent graders — deterministic, multi-seed, scores in [0.0, 1.0]."""
from __future__ import annotations
from typing import Protocol, Dict, Any, List
import numpy as np
from wellbeing_env import WellbeingFeedEnv
from models import WellbeingAction
 
 
class Agent(Protocol):
    def act(self, obs_vec: np.ndarray) -> int: ...
    def reset(self) -> None: ...
 
 
def _run_episode(env: WellbeingFeedEnv, agent: Agent) -> Dict[str, Any]:
    r = env.reset()
    obs = r.observation
    agent.reset()
    moods, engs, acts = [], [], []
    interventions = missed = 0
    while not env.done:
        v = obs.to_vector()
        a = agent.act(v)
        acts.append(a)
        if obs.user_mood < -0.1:
            interventions += 1 if a in (1, 2, 3) else 0
            missed += 1 if a == 0 else 0
        sr = env.step(WellbeingAction(action=a))
        obs = sr.observation
        moods.append(obs.user_mood)
        engs.append(obs.engagement_level)
    return {
        "score": env.get_episode_score(),
        "total_reward": env.total_reward,
        "avg_reward": env.total_reward / max(env.step_count, 1),
        "steps": env.step_count,
        "max_steps": env.task_config.max_steps,
        "completion": env.step_count / env.task_config.max_steps,
        "final_mood": moods[-1] if moods else 0.0,
        "min_mood": min(moods) if moods else 0.0,
        "avg_mood": float(np.mean(moods)) if moods else 0.0,
        "avg_eng": float(np.mean(engs)) if engs else 0.0,
        "interventions": interventions,
        "missed": missed,
        "action_dist": {i: acts.count(i) for i in range(5)},
    }
 
 
def grade_easy(agent: Agent, seeds: List[int] = None) -> Dict[str, Any]:
    seeds = seeds or [42, 123, 456]
    rs = [_run_episode(WellbeingFeedEnv("easy_stable_user", s), agent) for s in seeds]
    sc = float(np.mean([r["score"] for r in rs]))
    return {
        "task_id": "easy_stable_user", "difficulty": "easy",
        "avg_score": round(sc, 4), "passed": sc >= 0.6, "threshold": 0.6,
        "num_seeds": len(seeds), "per_seed": rs,
        "criteria": {
            "mood_maintained": float(np.mean([r["avg_mood"] for r in rs])) > 0.0,
            "engagement_maintained": float(np.mean([r["avg_eng"] for r in rs])) > 0.3,
            "completed": float(np.mean([r["completion"] for r in rs])) > 0.8,
        },
    }
 
 
def grade_medium(agent: Agent, seeds: List[int] = None) -> Dict[str, Any]:
    seeds = seeds or [42, 123, 456, 789, 1010]
    rs = [_run_episode(WellbeingFeedEnv("medium_spiral_detection", s), agent) for s in seeds]
    sc = float(np.mean([r["score"] for r in rs]))
    ai = float(np.mean([r["interventions"] for r in rs]))
    am = float(np.mean([r["missed"] for r in rs]))
    ir = ai / max(ai + am, 1) if ai else 0.0
    return {
        "task_id": "medium_spiral_detection", "difficulty": "medium",
        "avg_score": round(sc, 4), "passed": sc >= 0.5, "threshold": 0.5,
        "intervention_ratio": round(ir, 4), "num_seeds": len(seeds), "per_seed": rs,
        "criteria": {
            "spiral_detected": ir > 0.3,
            "mood_recovered": float(np.mean([r["final_mood"] for r in rs])) > -0.3,
            "not_abandoned": float(np.mean([r["completion"] for r in rs])) > 0.6,
        },
    }
 
 
def grade_hard(agent: Agent, seeds: List[int] = None) -> Dict[str, Any]:
    seeds = seeds or [42, 123, 456, 789, 1010, 2024, 3030]
    rs = [_run_episode(WellbeingFeedEnv("hard_engagement_trap", s), agent) for s in seeds]
    sc = float(np.mean([r["score"] for r in rs]))
    all_trending = all(r["action_dist"].get(0, 0) > r["steps"] * 0.7 for r in rs)
    return {
        "task_id": "hard_engagement_trap", "difficulty": "hard",
        "avg_score": round(sc, 4), "passed": sc >= 0.4, "threshold": 0.4,
        "num_seeds": len(seeds), "per_seed": rs,
        "criteria": {
            "not_pure_engagement": not all_trending,
            "mood_above_depression": float(np.mean([r["min_mood"] for r in rs])) > -0.8,
            "some_engagement": float(np.mean([r["avg_eng"] for r in rs])) > 0.2,
            "diverse_actions": not all_trending,
        },
    }
 
 
def grade_all(agent: Agent) -> Dict[str, Any]:
    e, m, h = grade_easy(agent), grade_medium(agent), grade_hard(agent)
    ov = e["avg_score"] * 0.25 + m["avg_score"] * 0.35 + h["avg_score"] * 0.40
    return {
        "overall_score": round(ov, 4),
        "all_passed": e["passed"] and m["passed"] and h["passed"],
        "tasks": {"easy": e, "medium": m, "hard": h},
    }