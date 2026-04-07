"""
Wellbeing Feed RL Environment — OpenEnv compatible.
step(action) → StepResult, reset() → ResetResult, state() → dict, close() → None.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any, List
from models import (
    WellbeingAction, WellbeingObservation, WellbeingReward,
    StepResult, ResetResult, ContentItem, UserProfile,
    TaskConfig, TASK_CONFIGS, ACTION_NAMES,
)
 
 
class WellbeingFeedEnv:
 
    def __init__(self, task_id: str = "easy_stable_user", seed: int = 42):
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task '{task_id}'. Available: {list(TASK_CONFIGS)}")
        self.task_config: TaskConfig = TASK_CONFIGS[task_id]
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.user: Optional[UserProfile] = None
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.done: bool = True
        self._log: List[dict] = []
        self._neg_cluster: int = 0
 
    # ── OpenEnv interface ──
 
    def reset(self, seed: Optional[int] = None) -> ResetResult:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        c = self.task_config
        self.user = UserProfile(
            mood=c.initial_mood, engagement=0.6, vulnerability=c.vulnerability,
            negativity_bias=c.negativity_bias, doomscroll_tendency=c.doomscroll_tendency,
            time_of_day=c.time_of_day,
        )
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self._log = []
        self._neg_cluster = 0
        return ResetResult(observation=self._obs(), done=False)
 
    def step(self, action: WellbeingAction) -> StepResult:
        if self.done:
            raise RuntimeError("Episode done. Call reset().")
        a = action.action
        c = self.task_config
 
        content = ContentItem.generate(a, self.rng)
        content = self._feed_dynamics(content)
        self._user_response(content)
 
        rew_obj = self._reward(content)
        self.total_reward += rew_obj.total
 
        self.step_count += 1
        self.user.session_time = min(self.step_count / c.max_steps, 1.0)
        self.user.time_of_day = min(c.time_of_day + self.step_count * 0.002, 1.0)
 
        self.done = (
            self.step_count >= c.max_steps
            or self.user.mood < -0.85
            or self.user.engagement < 0.05
        )
 
        obs = self._obs()
        info = {
            "step": self.step_count, "action": a,
            "action_name": ACTION_NAMES.get(a, "?"),
            "content_sentiment": round(content.sentiment, 4),
            "content_toxicity": round(content.toxicity, 4),
            "user_mood": round(self.user.mood, 4),
            "engagement": round(self.user.engagement, 4),
            "doomscroll": round(self.user.doomscroll_tendency, 4),
        }
        self._log.append(info)
        return StepResult(
            observation=obs, reward=rew_obj,
            done=self.done, info=info, last_action_error=None,
        )
 
    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_config.task_id,
            "difficulty": self.task_config.difficulty,
            "step_count": self.step_count,
            "max_steps": self.task_config.max_steps,
            "done": self.done,
            "total_reward": round(self.total_reward, 4),
            "avg_reward": round(self.total_reward / max(self.step_count, 1), 4),
            "user": {
                "mood": round(self.user.mood, 4),
                "engagement": round(self.user.engagement, 4),
                "vulnerability": round(self.user.effective_vulnerability, 4),
                "doomscroll": round(self.user.doomscroll_tendency, 4),
                "mood_trend": round(self.user.mood_trend, 4),
            } if self.user else None,
            "episode_log": self._log,
        }
 
    def close(self) -> None:
        pass
 
    @staticmethod
    def list_tasks() -> List[Dict[str, Any]]:
        return [
            {"task_id": c.task_id, "difficulty": c.difficulty,
             "max_steps": c.max_steps, "success_threshold": c.success_threshold}
            for c in TASK_CONFIGS.values()
        ]
 
    def get_episode_score(self) -> float:
        if self.step_count == 0:
            return 0.0
        avg = self.total_reward / self.step_count
        bonus = 0.1 if (self.user and self.user.mood > 0) else (
            0.05 if (self.user and self.user.mood > -0.3) else 0.0)
        comp = (self.step_count / self.task_config.max_steps) * 0.15
        return float(np.clip(avg + bonus + comp, 0.0, 1.0))
 
    # ── Internal dynamics ──
 
    def _obs(self) -> WellbeingObservation:
        u = self.user
        h = (u.content_history[-10:] + [0.0] * 10)[:10]
        return WellbeingObservation(
            user_mood=round(u.mood, 4), mood_trend=round(u.mood_trend, 4),
            engagement_level=round(u.engagement, 4),
            content_history=[round(x, 4) for x in h],
            session_duration=round(u.session_time, 4),
            vulnerability_score=round(u.effective_vulnerability, 4),
            scroll_velocity=round(u.scroll_velocity, 4),
            time_of_day=round(u.time_of_day, 4),
        )
 
    def _feed_dynamics(self, c: ContentItem) -> ContentItem:
        cfg = self.task_config
        if cfg.cluster_negative:
            if self._neg_cluster > 0:
                c.sentiment = min(c.sentiment, float(self.rng.uniform(-0.6, -0.1)))
                c.toxicity = max(c.toxicity, float(self.rng.uniform(0.3, 0.7)))
                self._neg_cluster -= 1
            elif self.rng.random() < 0.1:
                self._neg_cluster = int(self.rng.integers(3, 8))
        if cfg.engagement_trap and c.toxicity > 0.3:
            c.engagement_pull = min(c.engagement_pull + 0.25, 1.0)
        return c
 
    def _user_response(self, c: ContentItem) -> None:
        u = self.user
        mi = c.sentiment * 0.3
        if c.sentiment < 0:
            mi *= u.negativity_bias
        if mi < 0:
            mi *= (1.0 + u.effective_vulnerability)
        u.mood = float(np.clip(u.mood * u.mood_inertia + mi * (1.0 - u.mood_inertia), -1, 1))
        u.mood_history.append(u.mood)
 
        de = c.engagement_pull * 0.15 - u.engagement_decay
        if u.mood < -0.2 and u.doomscroll_tendency < 0.3:
            de -= 0.03 * abs(u.mood)
        if c.toxicity > 0.3 and u.doomscroll_tendency > 0.2:
            de += u.doomscroll_tendency * 0.1
        u.engagement = float(np.clip(u.engagement + de, 0, 1))
 
        if c.toxicity > 0.3 and c.engagement_pull > 0.5:
            u.doomscroll_tendency = min(u.doomscroll_tendency + 0.05, 1.0)
        elif c.sentiment > 0.3:
            u.doomscroll_tendency = max(u.doomscroll_tendency - 0.03, 0.0)
        u.content_history.append(c.sentiment)
 
    def _reward(self, c: ContentItem) -> WellbeingReward:
        u = self.user
        mood_h = (u.mood + 1.0) / 2.0
        imp = max(u.mood_trend * 0.5, 0.0)
        eng = u.engagement
        ad = 1.0 - u.doomscroll_tendency
        sr = 0.0
        if len(u.mood_history) >= 3:
            if (u.mood_history[-3] > u.mood_history[-2]
                    and u.mood > u.mood_history[-2]
                    and u.mood_history[-2] < 0):
                sr = 0.3
        tp = c.toxicity * 0.2
        raw = 0.30 * mood_h + 0.15 * imp + 0.25 * eng + 0.10 * ad + sr - tp
        total = float(np.clip(raw, 0.0, 1.0))
        return WellbeingReward(
            total=round(total, 4),
            components={
                "mood_health": round(mood_h, 4),
                "improvement": round(imp, 4),
                "engagement": round(eng, 4),
                "anti_doomscroll": round(ad, 4),
                "spiral_rescue": round(sr, 4),
                "toxicity_penalty": round(-tp, 4),
            },
        )
 