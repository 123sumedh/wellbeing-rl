"""Baseline agents: Random, Engagement-Only, Heuristic, Q-Learning."""
from __future__ import annotations
import numpy as np
from typing import Optional
from models import WellbeingAction
 
 
class RandomAgent:
    def __init__(self, seed=42): self.rng = np.random.default_rng(seed)
    def reset(self): pass
    def act(self, v: np.ndarray) -> int: return int(self.rng.integers(0, 5))
 
 
class EngagementOnlyAgent:
    """Always shows trending — demonstrates pure engagement failure."""
    def reset(self): pass
    def act(self, v: np.ndarray) -> int: return 0
 
 
class HeuristicAgent:
    """Rule-based wellbeing-aware policy."""
    def __init__(self): self._pm = 0.0
    def reset(self): self._pm = 0.0
    def act(self, v: np.ndarray) -> int:
        mood, trend, eng = v[0], v[1], v[2]
        vuln, scroll = v[4], v[5]
        if mood < -0.3: return 1
        if scroll > 0.6 and trend < -0.05: return 2
        if trend < -0.02 or mood < 0.0: return 3 if eng < 0.3 else 1
        if vuln > 0.5: return 2
        if eng < 0.35: return 4
        if mood > 0.3 and eng > 0.5: return 4
        return 3
 
 
class SimpleQLearningAgent:
    """Tabular Q-learning, 90 states x 5 actions."""
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.15, seed=42):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((90, 5))
        self._ps: Optional[int] = None
        self._pa: Optional[int] = None
        self.training = True
 
    def _disc(self, v):
        m = 0 if v[0]<-0.6 else 1 if v[0]<-0.2 else 2 if v[0]<0.2 else 3 if v[0]<0.6 else 4
        t = 0 if v[1]<-0.03 else 1 if v[1]<0.03 else 2
        e = 0 if v[2]<0.33 else 1 if v[2]<0.66 else 2
        vn = 0 if v[4]<0.4 else 1
        return m*18 + t*6 + e*2 + vn
 
    def reset(self): self._ps = self._pa = None
    def act(self, v):
        s = self._disc(v)
        a = (int(self.rng.integers(0, 5)) if self.training and self.rng.random() < self.epsilon
             else int(np.argmax(self.q_table[s])))
        self._ps, self._pa = s, a
        return a
 
    def learn(self, reward, nv, done):
        if self._ps is None: return
        ns = self._disc(nv)
        bn = 0.0 if done else float(np.max(self.q_table[ns]))
        self.q_table[self._ps, self._pa] += self.alpha * (
            reward + self.gamma * bn - self.q_table[self._ps, self._pa])
 
 
def wellbeing_agent(obs) -> WellbeingAction:
    """Functional wrapper for inference.py compatibility."""
    h = HeuristicAgent()
    v = obs.to_vector()
    return WellbeingAction(action=h.act(v))