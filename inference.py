#!/usr/bin/env python3
"""
Inference Script — Wellbeing Feed RL Environment
=================================================
MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

from wellbeing_env import WellbeingFeedEnv
from models import WellbeingAction, TASK_CONFIGS, ACTION_NAMES

# ─── Configuration ─────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "wellbeing-feed-rl"

SYSTEM_PROMPT = textwrap.dedent("""\
You are an RL agent controlling a social media feed recommendation system.
Your goal: balance user ENGAGEMENT with mental WELLBEING.

Each turn you see the user's state and must choose ONE action (0-4):
  0 = show_trending — High engagement but risky toxic content
  1 = show_motivational — Medium engagement, very positive for mood
  2 = show_educational — Low engagement, safe, mildly positive
  3 = show_funny — Medium-high engagement, good mood boost
  4 = show_personalized — High engagement, mixed sentiment

STRATEGY:
- user_mood < -0.2 or mood_trend < -0.03 -> action 1 or 3
- scroll_velocity > 0.6 and mood declining -> action 2 (pattern break)
- vulnerability_score > 0.5 -> avoid action 0
- engagement < 0.3 -> action 4 or 3
- mood > 0.3 and engagement > 0.5 -> action 4

Reply with ONLY a single digit (0-4). Nothing else.""")


# ─── Logging ───────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={rstr}", flush=True)


# ─── LLM Agent ─────────────────────────────────────────

_llm_failures = 0  # skip LLM after repeated failures

def get_llm_action(client: OpenAI, obs_dict: dict, step: int,
                    last_reward: float, history: List[str]) -> int:
    global _llm_failures
    if _llm_failures >= 5:
        return _heuristic(obs_dict)
    prompt = (
        f"Step: {step}\n"
        f"user_mood: {obs_dict['user_mood']:+.3f}  mood_trend: {obs_dict['mood_trend']:+.3f}\n"
        f"engagement: {obs_dict['engagement_level']:.3f}  "
        f"vulnerability: {obs_dict['vulnerability_score']:.3f}\n"
        f"scroll_velocity: {obs_dict['scroll_velocity']:.3f}  "
        f"session: {obs_dict['session_duration']:.0%}\n"
        f"recent_content: {obs_dict['content_history'][-5:]}\n"
        f"Last reward: {last_reward:.2f}\n"
        f"Choose action (0-4):"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2, max_tokens=5, stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()
        for ch in text:
            if ch in "01234":
                _llm_failures = 0
                return int(ch)
        return _heuristic(obs_dict)
    except KeyboardInterrupt:
        _llm_failures += 1
        return _heuristic(obs_dict)
    except Exception as exc:
        _llm_failures += 1
        if _llm_failures <= 3:
            print(f"[DEBUG] LLM error: {exc}", file=sys.stderr, flush=True)
        elif _llm_failures == 4:
            print(f"[DEBUG] LLM failed {_llm_failures}x, switching to heuristic fallback", file=sys.stderr, flush=True)
        return _heuristic(obs_dict)


def _heuristic(o: dict) -> int:
    mood, trend, eng = o.get("user_mood", 0), o.get("mood_trend", 0), o.get("engagement_level", 0.5)
    vuln, scroll = o.get("vulnerability_score", 0), o.get("scroll_velocity", 0)
    if mood < -0.3: return 1
    if scroll > 0.6 and trend < -0.05: return 2
    if trend < -0.02 or mood < 0: return 3 if eng < 0.3 else 1
    if vuln > 0.5: return 2
    if eng < 0.35: return 4
    return 3


# ─── Run one task ──────────────────────────────────────

def run_task(client: OpenAI, task_id: str, seed: int = 42) -> dict:
    cfg = TASK_CONFIGS[task_id]
    env = WellbeingFeedEnv(task_id=task_id, seed=seed)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        rr = env.reset()
        obs = rr.observation
        last_reward = 0.0

        for step_num in range(1, cfg.max_steps + 1):
            if env.done:
                break
            obs_dict = obs.model_dump()
            action_int = get_llm_action(client, obs_dict, step_num, last_reward, history)
            action_name = ACTION_NAMES.get(action_int, f"action_{action_int}")

            result = env.step(WellbeingAction(action=action_int))
            reward_val = result.reward.total
            rewards.append(reward_val)
            steps_taken = step_num
            last_reward = reward_val
            obs = result.observation

            log_step(step=step_num, action=action_name, reward=reward_val,
                     done=result.done, error=result.last_action_error)

            history.append(f"Step {step_num}: {action_name} mood={obs.user_mood:+.3f} "
                           f"eng={obs.engagement_level:.3f} r={reward_val:.2f}")
            if result.done:
                break

        score = min(max(env.get_episode_score(), 0.0), 1.0)
        success = score >= cfg.success_threshold

    except (Exception, KeyboardInterrupt) as exc:
        print(f"[DEBUG] task error: {exc}", file=sys.stderr, flush=True)
        if steps_taken > 0:
            score = min(max(env.get_episode_score(), 0.0), 1.0)
            success = score >= cfg.success_threshold

    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": round(score, 4), "passed": success,
            "threshold": cfg.success_threshold, "steps": steps_taken}


# ─── Main ──────────────────────────────────────────────

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_ids = list(TASK_CONFIGS.keys())
    global _llm_failures

    results = []
    for task_id in task_ids:
        _llm_failures = 0  # reset per task
        try:
            result = run_task(client, task_id, seed=42)
        except (Exception, KeyboardInterrupt) as exc:
            print(f"[DEBUG] Fatal error on {task_id}: {exc}", file=sys.stderr, flush=True)
            result = {"task_id": task_id, "score": 0.0, "passed": False,
                      "threshold": TASK_CONFIGS[task_id].success_threshold, "steps": 0}
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {task_id}: {result['score']:.3f} "
              f"(>={result['threshold']})", flush=True)

    avg = sum(r["score"] for r in results) / len(results)
    print(f"\nOverall: {avg:.4f}  All passed: {all(r['passed'] for r in results)}", flush=True)


if __name__ == "__main__":
    main()