"""Scene agent: answer natural-language questions about a processed video.

The LLM never sees the video directly. It plans which tools to call; numeric
tools query what the tracker logged (exact counts, times, occupancy), and the
`look_at_frame` tool sends a saved keyframe to a vision-language model for
visual questions the tracker cannot answer (clothing, actions, scene context).

    # one question
    python -m epa.agent --run results/site -q "Which workers entered the site without a hard hat?"
    # interactive
    python -m epa.agent --run results/site
"""

from __future__ import annotations

import argparse
import json
from typing import Callable

from epa.llm import ChatClient
from epa.scene_store import SceneStore

SYSTEM_PROMPT = """You are a video analytics assistant. You answer questions about ONE \
recorded video that has already been processed by an object detector and tracker.

Rules:
- Use the tools. Never guess a number: every count, time or ID in your answer must \
come from a tool result in this conversation.
- Times are in seconds from the start of the video.
- For visual details the tracker does not record (appearance, clothing, actions, \
objects it does not detect), call look_at_frame.
- Object classes use the detector's names, e.g. "person", "car", "bus", "bicycle".
- Safety runs (get_video_summary shows "safety"): violations are raised by the detector \
when a worker without a hard hat (rule helmet_required; "headwear": "other_headwear" \
means a cap, hat or non-work helmet instead of a hard hat), without a high-visibility \
vest (vest_required) or without a mask (mask_required) stays in a zone that requires it; \
machine_proximity = a worker closer than about one body height to a machine or vehicle \
(machine_moving says whether it was moving; distances come from a single camera, so they \
are approximate); cone_area_entry = a person (or vehicle) inside the area marked by cones. Use \
inspect_violation to get a second opinion from the vision model before calling a \
violation confirmed, and report disagreements instead of hiding them. Violations \
with evidence_type "inferred_from_absence" (no helmet / vest was found, rather than a \
bare head being detected) are the ones that most need this check; every vest \
violation is of this type.
- If the tools cannot answer the question, say so plainly.
- Answer in the user's language, briefly, and mention the times you relied on."""


def _tool(name: str, description: str, properties: dict, required: list[str] | None = None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required or []}}}


TOOLS = [
    _tool("get_video_summary",
          "Video duration, zones and their class filters, totals per zone, unique tracks per class. "
          "Call this first if you do not know the zone names.", {}),
    _tool("count_events",
          "Count zone enter/exit events, optionally filtered by zone, event type, class and time window.",
          {"zone": {"type": "string"},
           "event": {"type": "string", "enum": ["enter", "exit"]},
           "object_class": {"type": "string"},
           "t_start": {"type": "number", "description": "seconds, inclusive"},
           "t_end": {"type": "number", "description": "seconds, inclusive"}}),
    _tool("zone_occupancy",
          "Objects inside a zone at time t, or (without t) peak, mean and fraction of time occupied.",
          {"zone": {"type": "string"}, "t": {"type": "number"}}, ["zone"]),
    _tool("objects_at",
          "All tracked objects visible at time t, with IDs, classes and boxes.",
          {"t": {"type": "number"}}, ["t"]),
    _tool("track_history",
          "When one tracked object was first and last seen, and its zone events.",
          {"track_id": {"type": "integer"}}, ["track_id"]),
    _tool("list_violations",
          "Safety violations (missing hard hat / vest / mask, machine proximity, coned-off area entry), "
          "optionally filtered by zone and time window.",
          {"zone": {"type": "string"},
           "t_start": {"type": "number"}, "t_end": {"type": "number"}}),
    _tool("ppe_status_at",
          "Helmet status (helmet / no_helmet / unknown) and, if vests are checked, vest status "
          "(vest / no_vest / unknown) of every worker visible at time t.",
          {"t": {"type": "number"}}, ["t"]),
    _tool("inspect_violation",
          "Send the cropped evidence image of one violation to the vision model with a question, "
          "e.g. to verify whether the worker is really not wearing a hard hat.",
          {"violation_id": {"type": "integer"}, "question": {"type": "string"}},
          ["violation_id"]),
    _tool("look_at_frame",
          "Ask a vision-language model a visual question about the saved frame closest to time t.",
          {"t": {"type": "number"}, "question": {"type": "string"}}, ["t", "question"]),
]


class SceneAgent:
    def __init__(self, store: SceneStore, client: ChatClient, max_steps: int = 8,
                 verbose: bool = False, log: Callable[[str], None] = print):
        self.store = store
        self.client = client
        self.max_steps = max_steps
        self.verbose = verbose
        self.log = log
        self.trace: list[dict] = []

    # ------------------------------------------------------------ tool calls
    def _look_at_frame(self, t: float, question: str) -> dict:
        kf = self.store.keyframe_near(t)
        if kf is None:
            return {"error": "no keyframes saved for this run"}
        answer = self.client.describe_image(kf["path"], question)
        return {"frame_t": kf["t"], "requested_t": t, "answer": answer}

    def _inspect_violation(self, vid: int, question: str | None) -> dict:
        ev = self.store.violation_evidence(vid)
        if "error" in ev:
            return ev
        rule = ev.get("rule")
        if rule == "machine_proximity":
            default_q = ("Is a worker standing dangerously close to the machine or vehicle in this "
                         "image (within about one body length, on the ground)?")
        elif rule == "cone_area_entry":
            default_q = "Is the person inside the area marked off by traffic / safety cones?"
        else:
            item = {"vest_required": "a high-visibility safety vest",
                    "mask_required": "a face mask"}.get(rule, "a construction hard hat (not a cap, "
                                                              "hat or bicycle / motorcycle helmet)")
            default_q = f"Is the person in this image wearing {item}?"
        q = question or (default_q + " Answer 'yes', 'no' or 'unclear', then one short sentence "
                         "of evidence.")
        return {"violation_id": vid, "t": ev["t"], "track_id": ev["track_id"],
                "answer": self.client.describe_image(ev["path"], q)}

    def execute(self, name: str, args: dict) -> dict:
        s = self.store
        try:
            if name == "get_video_summary":
                return s.get_video_summary()
            if name == "count_events":
                return s.count_events(**{k: args.get(k) for k in
                                         ("zone", "event", "object_class", "t_start", "t_end")})
            if name == "zone_occupancy":
                return s.zone_occupancy(args["zone"], args.get("t"))
            if name == "objects_at":
                return s.objects_at(float(args["t"]))
            if name == "track_history":
                return s.track_history(int(args["track_id"]))
            if name == "list_violations":
                return s.list_violations(args.get("zone"), args.get("t_start"), args.get("t_end"))
            if name == "ppe_status_at":
                return s.ppe_status_at(float(args["t"]))
            if name == "inspect_violation":
                return self._inspect_violation(int(args["violation_id"]), args.get("question"))
            if name == "look_at_frame":
                return self._look_at_frame(float(args["t"]), str(args["question"]))
            return {"error": f"unknown tool {name}"}
        except (KeyError, TypeError, ValueError) as e:
            return {"error": f"bad arguments for {name}: {e}"}

    # ------------------------------------------------------------- main loop
    def ask(self, question: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}]
        self.trace = []
        for _ in range(self.max_steps):
            msg = self.client.chat(messages, TOOLS)
            calls = msg.get("tool_calls") or []
            if not calls:
                return (msg.get("content") or "").strip()
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": calls})
            for call in calls:
                fn = call["function"]
                args = fn.get("arguments") or {}
                if isinstance(args, str):           # OpenAI sends a JSON string,
                    try:                             # some local servers send a dict
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {}
                result = self.execute(fn["name"], args)
                self.trace.append({"tool": fn["name"], "args": args, "result": result})
                if self.verbose:
                    self.log(f"  -> {fn['name']}({json.dumps(args)}) = "
                             f"{json.dumps(result)[:200]}")
                messages.append({"role": "tool", "tool_call_id": call.get("id", fn["name"]),
                                 "name": fn["name"],
                                 "content": json.dumps(result, ensure_ascii=False)})
        return "I could not finish the analysis within the step limit."


def main():
    p = argparse.ArgumentParser(description="Ask questions about a processed video")
    p.add_argument("--run", required=True, help="output folder of epa.pipeline")
    p.add_argument("-q", "--question", help="one question; omit for interactive mode")
    p.add_argument("--model", help="LLM for tool calling (default env EPA_LLM_MODEL)")
    p.add_argument("--vlm", help="vision model (default env EPA_VLM_MODEL)")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint (default: local Ollama)")
    p.add_argument("-v", "--verbose", action="store_true", help="print tool calls")
    a = p.parse_args()

    agent = SceneAgent(SceneStore(a.run),
                       ChatClient(base_url=a.base_url, model=a.model, vlm_model=a.vlm),
                       verbose=a.verbose)
    if a.question:
        print(agent.ask(a.question))
        return
    print("Ask about the video (empty line to quit).")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            break
        print(agent.ask(q))


if __name__ == "__main__":
    main()
