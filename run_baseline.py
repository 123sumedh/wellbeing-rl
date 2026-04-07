#!/usr/bin/env python3
"""Local baseline evaluation. Reproducible scores, no LLM needed."""
import argparse, sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wellbeing_env import WellbeingFeedEnv
from models import WellbeingAction, TASK_CONFIGS
from baseline_agents import RandomAgent, HeuristicAgent, EngagementOnlyAgent, SimpleQLearningAgent
from graders import grade_all
 
def train_q_agent(agent, task_id, num_episodes=500, seed=42):
    rng = np.random.default_rng(seed)
    rewards = []
    for ep in range(num_episodes):
        env = WellbeingFeedEnv(task_id, int(rng.integers(0, 100000)))
        r = env.reset(); obs = r.observation
        agent.reset(); agent.training = True
        agent.epsilon = max(0.05, 0.3 * (1 - ep / num_episodes))
        total = 0.0
        while not env.done:
            v = obs.to_vector(); a = agent.act(v)
            sr = env.step(WellbeingAction(action=a))
            agent.learn(sr.reward.total, sr.observation.to_vector(), sr.done)
            obs = sr.observation; total += sr.reward.total
        rewards.append(total)
    agent.training = False; agent.epsilon = 0.0
    return rewards
 
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--train-episodes", type=int, default=500)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--output", type=str, default=None)
    a = pa.parse_args()
    print(f"Wellbeing Feed RL — Baseline Benchmark (seed={a.seed})\n")
    res = {}
    for nm, ag in [("random", RandomAgent(a.seed)), ("engagement_only", EngagementOnlyAgent()),
                    ("heuristic", HeuristicAgent())]:
        r = grade_all(ag); res[nm] = r
        print(f"  {nm:20s} overall={r['overall_score']:.4f} passed={r['all_passed']}")
    print(f"\n  Training Q-Learning ({a.train_episodes} ep)...")
    q = SimpleQLearningAgent(seed=a.seed)
    for tid in TASK_CONFIGS:
        h = train_q_agent(q, tid, a.train_episodes, a.seed)
        print(f"    {tid}: last-50 avg={np.mean(h[-50:]):.3f}")
    r = grade_all(q); res["q_learning"] = r
    print(f"  {'q_learning':20s} overall={r['overall_score']:.4f} passed={r['all_passed']}")
    print(f"\n{'='*55}\nSUMMARY")
    for n, r in res.items():
        s = r["overall_score"]
        print(f"  {n:20s} {'#'*int(s*40)}{'.'*(40-int(s*40))} {s:.4f}")
    if a.output:
        def san(o):
            if isinstance(o,(np.float32,np.float64)):return float(o)
            if isinstance(o,(np.int32,np.int64)):return int(o)
            if isinstance(o,np.ndarray):return o.tolist()
            if isinstance(o,dict):return{k:san(v)for k,v in o.items()}
            if isinstance(o,list):return[san(v)for v in o]
            return o
        json.dump(san(res), open(a.output,"w"), indent=2)
        print(f"Saved to {a.output}")
 
if __name__ == "__main__":
    main()