"""
Typed Pydantic models for the Wellbeing Feed RL Environment.
OpenEnv spec: Observation, Action, Reward with full validation.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
 
 
class WellbeingAction(BaseModel):
    """Typed action: choose one of 5 content categories."""
    action: int = Field(
        ..., ge=0, le=4,
        description="0=trending 1=motivational 2=educational 3=funny 4=personalized",
    )
 
 
class WellbeingObservation(BaseModel):
    """What the agent sees — user's psychological state + feed context."""
    user_mood: float = Field(..., ge=-1.0, le=1.0)
    mood_trend: float = Field(..., ge=-1.0, le=1.0)
    engagement_level: float = Field(..., ge=0.0, le=1.0)
    content_history: List[float] = Field(..., min_length=10, max_length=10)
    session_duration: float = Field(..., ge=0.0, le=1.0)
    vulnerability_score: float = Field(..., ge=0.0, le=1.0)
    scroll_velocity: float = Field(..., ge=0.0, le=1.0)
    time_of_day: float = Field(..., ge=0.0, le=1.0)
 
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.user_mood, self.mood_trend, self.engagement_level,
            self.session_duration, self.vulnerability_score,
            self.scroll_velocity, self.time_of_day,
        ] + self.content_history, dtype=np.float32)
 
    model_config = {"arbitrary_types_allowed": True}
 
 
class WellbeingReward(BaseModel):
    """Reward with component breakdown."""
    total: float = Field(..., ge=0.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
 
 
class StepResult(BaseModel):
    """Returned by env.step()."""
    observation: WellbeingObservation
    reward: WellbeingReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    last_action_error: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}
 
 
class ResetResult(BaseModel):
    """Returned by env.reset()."""
    observation: WellbeingObservation
    done: bool = False
    model_config = {"arbitrary_types_allowed": True}
 
 
# ── Internal simulation types ──
 
class ContentItem:
    __slots__ = ("category", "sentiment", "engagement_pull", "virality", "toxicity")
    def __init__(self, category, sentiment, engagement_pull, virality, toxicity):
        self.category = category
        self.sentiment = sentiment
        self.engagement_pull = engagement_pull
        self.virality = virality
        self.toxicity = toxicity
 
    @staticmethod
    def generate(category: int, rng: np.random.Generator) -> ContentItem:
        P = {
            0: (-0.1, 0.40, 0.75, 0.35),   # trending
            1: (0.70, 0.15, 0.45, 0.02),    # motivational
            2: (0.30, 0.20, 0.35, 0.05),    # educational
            3: (0.50, 0.25, 0.65, 0.08),    # funny
            4: (0.20, 0.30, 0.70, 0.15),    # personalized
        }
        sm, ss, em, tm = P[category]
        return ContentItem(
            category=category,
            sentiment=float(np.clip(rng.normal(sm, ss), -1, 1)),
            engagement_pull=float(np.clip(rng.normal(em, 0.15), 0, 1)),
            virality=float(np.clip(rng.beta(2, 5), 0, 1)),
            toxicity=float(np.clip(rng.exponential(tm), 0, 1)),
        )
 
 
class UserProfile:
    """Simulated user with realistic psychological dynamics."""
    def __init__(self, mood=0.3, engagement=0.5, vulnerability=0.2,
                 mood_inertia=0.85, engagement_decay=0.02, negativity_bias=1.4,
                 doomscroll_tendency=0.0, session_time=0.0, time_of_day=0.5):
        self.mood = mood
        self.engagement = engagement
        self.vulnerability = vulnerability
        self.mood_inertia = mood_inertia
        self.engagement_decay = engagement_decay
        self.negativity_bias = negativity_bias
        self.doomscroll_tendency = doomscroll_tendency
        self.session_time = session_time
        self.time_of_day = time_of_day
        self.mood_history: List[float] = [mood]
        self.content_history: List[float] = []
 
    @property
    def mood_trend(self) -> float:
        if len(self.mood_history) < 2:
            return 0.0
        recent = self.mood_history[-5:]
        diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        return float(np.clip(np.mean(diffs), -1, 1))
 
    @property
    def scroll_velocity(self) -> float:
        v = 0.3
        if self.engagement < 0.3:
            v += 0.4
        v += self.doomscroll_tendency * 0.3
        return float(np.clip(v, 0, 1))
 
    @property
    def effective_vulnerability(self) -> float:
        tf = max(0.0, (self.time_of_day - 0.7) / 0.3 * 0.3) if self.time_of_day > 0.7 else 0.0
        return float(np.clip(self.vulnerability + tf + self.session_time * 0.2, 0, 1))
 
 
# ── Task configs ──
 
class TaskConfig(BaseModel):
    task_id: str
    difficulty: str
    max_steps: int
    success_threshold: float
    initial_mood: float = 0.3
    vulnerability: float = 0.2
    negativity_bias: float = 1.4
    doomscroll_tendency: float = 0.0
    time_of_day: float = 0.5
    negative_content_probability: float = 0.2
    cluster_negative: bool = False
    engagement_trap: bool = False
 
 
TASK_CONFIGS: Dict[str, TaskConfig] = {
    "easy_stable_user": TaskConfig(
        task_id="easy_stable_user", difficulty="easy", max_steps=50,
        success_threshold=0.6, initial_mood=0.4, vulnerability=0.15,
        negativity_bias=1.2, time_of_day=0.4,
    ),
    "medium_spiral_detection": TaskConfig(
        task_id="medium_spiral_detection", difficulty="medium", max_steps=100,
        success_threshold=0.5, initial_mood=0.2, vulnerability=0.45,
        negativity_bias=1.6, doomscroll_tendency=0.1, time_of_day=0.6,
        negative_content_probability=0.35, cluster_negative=True,
    ),
    "hard_engagement_trap": TaskConfig(
        task_id="hard_engagement_trap", difficulty="hard", max_steps=150,
        success_threshold=0.4, initial_mood=0.1, vulnerability=0.6,
        negativity_bias=1.8, doomscroll_tendency=0.4, time_of_day=0.85,
        negative_content_probability=0.45, cluster_negative=True,
        engagement_trap=True,
    ),
}
 
ACTION_NAMES: Dict[int, str] = {
    0: "show_trending",
    1: "show_motivational",
    2: "show_educational",
    3: "show_funny",
    4: "show_personalized",
}