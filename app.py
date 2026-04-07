#!/usr/bin/env python3
"""
HF Spaces App — Gradio UI + FastAPI endpoints.
POST /reset returns 200 (validator ping).
"""
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request
import numpy as np
import sys, os
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from wellbeing_env import WellbeingFeedEnv
from models import WellbeingAction, TASK_CONFIGS, ACTION_NAMES
from baseline_agents import RandomAgent, HeuristicAgent, EngagementOnlyAgent, SimpleQLearningAgent
from graders import grade_all
 
# ═══ FastAPI app ═══
fapi = FastAPI()
_ENVS: dict = {}
 
@fapi.post("/reset")
async def api_reset(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    tid = body.get("task_id", "easy_stable_user")
    seed = body.get("seed", 42)
    if tid not in TASK_CONFIGS:
        tid = "easy_stable_user"
    env = WellbeingFeedEnv(task_id=tid, seed=int(seed))
    r = env.reset()
    _ENVS[tid] = env
    return JSONResponse({"observation": r.observation.model_dump(),
                         "done": r.done, "task_id": tid,
                         "max_steps": env.task_config.max_steps})
 
@fapi.post("/step")
async def api_step(request: Request):
    body = await request.json()
    tid = body.get("task_id", "easy_stable_user")
    action = body.get("action", 1)
    env = _ENVS.get(tid)
    if not env:
        return JSONResponse({"error": f"No session for '{tid}'. POST /reset first."}, 400)
    if env.done:
        return JSONResponse({"error": "Episode done.", "score": env.get_episode_score()})
    r = env.step(WellbeingAction(action=int(action)))
    out = {"observation": r.observation.model_dump(), "reward": r.reward.model_dump(),
           "done": r.done, "info": r.info}
    if r.done:
        out["score"] = env.get_episode_score()
    return JSONResponse(out)
 
@fapi.get("/state")
async def api_state(task_id: str = "easy_stable_user"):
    env = _ENVS.get(task_id)
    if not env:
        return JSONResponse({"error": f"No session for '{task_id}'."}, 400)
    return JSONResponse(env.state())
 
@fapi.get("/tasks")
async def api_tasks():
    return JSONResponse(WellbeingFeedEnv.list_tasks())
 
# ═══ Benchmark ═══
def run_benchmark(train_episodes: int, seed: int) -> str:
    from run_baseline import train_q_agent
    res = {}
    res["Random"] = grade_all(RandomAgent(int(seed)))
    res["Engagement-Only"] = grade_all(EngagementOnlyAgent())
    res["Heuristic"] = grade_all(HeuristicAgent())
    q = SimpleQLearningAgent(seed=int(seed))
    for tid in TASK_CONFIGS:
        train_q_agent(q, tid, int(train_episodes), int(seed))
    res["Q-Learning"] = grade_all(q)
 
    lines = ["# Benchmark Results\n",
             "| Agent | Easy | Medium | Hard | Overall |",
             "|-------|------|--------|------|---------|"]
    for nm, r in res.items():
        e, m, h = r["tasks"]["easy"], r["tasks"]["medium"], r["tasks"]["hard"]
        lines.append(f"| {nm} | {e['avg_score']:.3f} {'✓' if e['passed'] else '✗'} | "
                     f"{m['avg_score']:.3f} {'✓' if m['passed'] else '✗'} | "
                     f"{h['avg_score']:.3f} {'✓' if h['passed'] else '✗'} | "
                     f"**{r['overall_score']:.3f}** |")
    lines.append("\nEngagement-Only scores lowest — pure click optimization destroys wellbeing.")
    return "\n".join(lines)
 
# ═══ Interactive Play ═══
_P = {"env": None, "h": []}
def play_start(tid, seed):
    env = WellbeingFeedEnv(tid, int(seed)); r = env.reset()
    _P["env"], _P["h"] = env, []
    return _fobs(r.observation, 0), _ftraj([]), "Started!", gr.update(interactive=True)
 
def play_action(name):
    env = _P["env"]
    if not env or env.done:
        return "—", _ftraj(_P["h"]), "Done." if env and env.done else "Start first.", gr.update(interactive=False)
    amap = {"🔥 Trending":0,"💪 Motivational":1,"📚 Educational":2,"😂 Funny":3,"❤️ Personalized":4}
    r = env.step(WellbeingAction(action=amap.get(name, 0)))
    _P["h"].append({"s": env.step_count, "a": amap.get(name,0),
                     "m": r.observation.user_mood, "e": r.observation.engagement_level,
                     "r": r.reward.total})
    st = ""
    if r.done:
        sc = env.get_episode_score()
        st = f"**Done!** Score: {sc:.3f} {'✅ PASS' if sc >= env.task_config.success_threshold else '❌ FAIL'}"
    return (_fobs(r.observation, env.step_count), _ftraj(_P["h"]),
            st or f"Step {env.step_count} — Reward: {r.reward.total:.3f}",
            gr.update(interactive=not r.done))
 
def _fobs(o, step):
    e = "😊" if o.user_mood>0.3 else "😐" if o.user_mood>-0.2 else "😢" if o.user_mood>-0.5 else "😰"
    return (f"### Step {step}\n| Signal | Value |\n|---|---|\n"
            f"| {e} Mood | {o.user_mood:+.3f} (trend {o.mood_trend:+.3f}) |\n"
            f"| 📱 Engagement | {o.engagement_level:.3f} |\n"
            f"| ⚡ Scroll | {o.scroll_velocity:.3f} |\n"
            f"| 🎯 Vulnerability | {o.vulnerability_score:.3f} |\n"
            f"| ⏱️ Session | {o.session_duration:.0%} |")
 
def _ftraj(h):
    if not h: return "Take actions to see trajectory."
    lines = ["| Step | Action | Mood | Eng | Reward |", "|---|---|---|---|---|"]
    for x in h[-12:]:
        lines.append(f"| {x['s']} | {ACTION_NAMES.get(x['a'],'?')} | {x['m']:+.3f} | {x['e']:.3f} | {x['r']:.3f} |")
    if h:
        lines.append(f"\n**Avg:** mood={np.mean([x['m'] for x in h]):+.3f}  reward={np.mean([x['r'] for x in h]):.3f}")
    return "\n".join(lines)
 
# ═══ Gradio Blocks ═══
with gr.Blocks(title="Wellbeing Feed RL", theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown("# 🧠 Wellbeing Feed RL Environment\n*Balancing Engagement & Mental Health*")
    with gr.Tabs():
        with gr.TabItem("📖 About"):
            gr.Markdown("""## Problem
Social media algorithms optimize **engagement** — but engaging content is often toxic.
This creates doomscroll spirals that damage mental health.
 
## Challenge
Build an RL agent balancing **engagement** (keep user) with **wellbeing** (protect mood).
 
## 5 Actions
| ID | Category | Engagement | Mood | Risk |
|----|----------|-----------|------|------|
| 0 | Trending | High | Negative | High |
| 1 | Motivational | Medium | Very Positive | Low |
| 2 | Educational | Low | Mild Positive | Very Low |
| 3 | Funny | Med-High | Positive | Low |
| 4 | Personalized | High | Mixed | Medium |
 
## 8 Observation Signals
user_mood, mood_trend, engagement_level, content_history (10),
session_duration, vulnerability_score, scroll_velocity, time_of_day
 
## 3 Tasks
| Task | Steps | Threshold | Challenge |
|------|-------|-----------|-----------|
| easy_stable_user | 50 | ≥0.6 | Maintain mood + engagement |
| medium_spiral_detection | 100 | ≥0.5 | Detect/break negative spirals |
| hard_engagement_trap | 150 | ≥0.4 | Break doomscroll without losing user |
 
## Reward (6 components, 0-1)
30% mood health + 25% engagement + 15% mood improvement +
10% anti-doomscroll + spiral rescue bonus - toxicity penalty""")
 
        with gr.TabItem("📊 Benchmark"):
            with gr.Row():
                tr_eps = gr.Slider(50, 2000, 300, step=50, label="Q-Learning Episodes")
                b_seed = gr.Number(42, label="Seed", precision=0)
            b_btn = gr.Button("🚀 Run Benchmark", variant="primary")
            b_out = gr.Markdown("Click to start...")
            b_btn.click(run_benchmark, [tr_eps, b_seed], b_out)
 
        with gr.TabItem("🎮 Play"):
            gr.Markdown("**You** are the recommendation agent. Choose content for each reel.")
            with gr.Row():
                t_sel = gr.Dropdown(list(TASK_CONFIGS), value="easy_stable_user", label="Task")
                p_seed = gr.Number(42, label="Seed", precision=0)
                s_btn = gr.Button("▶️ Start", variant="primary")
            with gr.Row():
                with gr.Column():
                    o_disp = gr.Markdown("Click Start.")
                    st_disp = gr.Markdown("")
                    with gr.Row():
                        btns = {n: gr.Button(n) for n in
                                ["🔥 Trending","💪 Motivational","📚 Educational","😂 Funny","❤️ Personalized"]}
                with gr.Column():
                    tr_disp = gr.Markdown("No data.")
            first_btn = list(btns.values())[0]
            s_btn.click(play_start, [t_sel, p_seed], [o_disp, tr_disp, st_disp, first_btn])
            for nm, bt in btns.items():
                bt.click(play_action, gr.State(nm), [o_disp, tr_disp, st_disp, first_btn])
 
        with gr.TabItem("🔌 API"):
            gr.Markdown("## Endpoints\n`POST /reset` `POST /step` `GET /state` `GET /tasks`\n\n"
                        "The validator pings `POST /reset` — returns 200 with observation.")
 
# Mount Gradio inside FastAPI
app = gr.mount_gradio_app(fapi, demo, path="/")
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)