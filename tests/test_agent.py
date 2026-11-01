"""Scene agent tests — no real model needed.

- SceneStore queries on a small synthetic run
- The agent loop driven by a scripted fake LLM (tool call -> tool result -> answer)
- The HTTP payloads ChatClient sends, checked with httpx.MockTransport
"""

import json

import httpx
import pytest

from epa.agent import TOOLS, SceneAgent
from epa.llm import ChatClient
from epa.scene_store import SceneStore


@pytest.fixture
def run_dir(tmp_path):
    (tmp_path / "keyframes").mkdir()
    (tmp_path / "keyframes" / "frame_000010.jpg").write_bytes(b"\xff\xd8fakejpeg")
    (tmp_path / "keyframes" / "index.json").write_text(json.dumps(
        [{"frame": 10, "t": 1.0, "file": "frame_000010.jpg", "reason": "event"}]))
    (tmp_path / "summary.json").write_text(json.dumps({
        "frames": 30, "fps_source": 10, "duration_s": 3.0, "events": 3,
        "tracks_per_class": {"person": 2, "car": 1},
        "zones": [{"zone": "door", "occupancy": 0, "entries": 2, "exits": 1}],
        "zone_definitions": [{"name": "door", "classes": None}]}))
    events = [
        {"frame": 10, "t": 1.0, "zone": "door", "event": "enter", "track_id": 1, "class": "person"},
        {"frame": 15, "t": 1.5, "zone": "door", "event": "enter", "track_id": 3, "class": "car"},
        {"frame": 25, "t": 2.5, "zone": "door", "event": "exit", "track_id": 1, "class": "person"},
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    frames = []
    for i in range(30):
        t = i / 10
        tracks = [{"id": 1, "class": "person", "score": 0.9, "box": [0, 0, 10, 10]}]
        if i >= 15:
            tracks.append({"id": 3, "class": "car", "score": 0.8, "box": [5, 5, 20, 20]})
        occ = (1 if 10 <= i < 25 else 0) + (1 if i >= 15 else 0)
        frames.append({"frame": i, "t": t, "tracks": tracks, "occupancy": {"door": occ}})
    (tmp_path / "tracks.jsonl").write_text("\n".join(json.dumps(f) for f in frames))
    return tmp_path


# ------------------------------------------------------------------ store
def test_count_events_filters(run_dir):
    s = SceneStore(run_dir)
    assert s.count_events()["count"] == 3
    assert s.count_events(event="enter")["count"] == 2
    assert s.count_events(event="enter", object_class="person")["count"] == 1
    assert s.count_events(t_start=1.2)["count"] == 2
    assert "error" in s.count_events(zone="nope")


def test_occupancy_and_objects(run_dir):
    s = SceneStore(run_dir)
    assert s.zone_occupancy("door", t=2.0)["occupancy"] == 2
    peak = s.zone_occupancy("door")
    assert peak["peak_occupancy"] == 2 and peak["peak_at_t"] == 1.5
    assert s.objects_at(0.52)["counts"] == {"person": 1}
    assert s.objects_at(2.0)["counts"] == {"person": 1, "car": 1}


def test_track_history(run_dir):
    h = SceneStore(run_dir).track_history(3)
    assert h["class"] == "car" and h["first_seen_t"] == 1.5
    assert [e["event"] for e in h["events"]] == ["enter"]


# ------------------------------------------------------------- agent loop
class FakeClient:
    """Plays back a fixed sequence of assistant messages and records requests."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.vision_calls = []

    def chat(self, messages, tools=None):
        self.requests.append(json.loads(json.dumps(messages)))
        return self.script.pop(0)

    def describe_image(self, path, question):
        self.vision_calls.append((path, question))
        return "A person in a red jacket walking through the door."


def _call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def test_agent_tool_loop(run_dir):
    fake = FakeClient([
        # OpenAI style: arguments as a JSON string
        {"role": "assistant", "content": "", "tool_calls": [
            _call("c1", "count_events", json.dumps({"event": "enter", "object_class": "person"}))]},
        # Ollama style: arguments as a dict
        {"role": "assistant", "content": "", "tool_calls": [
            _call("c2", "look_at_frame", {"t": 1.1, "question": "What is the person wearing?"})]},
        {"role": "assistant", "content": "1 person entered the door at 1.0 s, wearing a red jacket."},
    ])
    agent = SceneAgent(SceneStore(run_dir), fake)
    answer = agent.ask("Who entered the door and what were they wearing?")

    assert answer.startswith("1 person entered")
    assert [t["tool"] for t in agent.trace] == ["count_events", "look_at_frame"]
    assert agent.trace[0]["result"]["count"] == 1
    assert agent.trace[1]["result"]["frame_t"] == 1.0
    assert fake.vision_calls[0][0].endswith("frame_000010.jpg")
    # the tool result is fed back to the model with the matching call id
    last_req = fake.requests[-1]
    tool_msgs = [m for m in last_req if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]


def test_agent_survives_bad_arguments(run_dir):
    fake = FakeClient([
        {"role": "assistant", "content": "", "tool_calls": [_call("c1", "objects_at", "{not json")]},
        {"role": "assistant", "content": "I could not determine that."},
    ])
    agent = SceneAgent(SceneStore(run_dir), fake)
    assert agent.ask("What is there?") == "I could not determine that."
    assert "error" in agent.trace[0]["result"]


def test_agent_step_limit(run_dir):
    loop = {"role": "assistant", "content": "", "tool_calls": [_call("c", "get_video_summary", "{}")]}
    agent = SceneAgent(SceneStore(run_dir), FakeClient([loop] * 3), max_steps=3)
    assert "step limit" in agent.ask("?")


# ----------------------------------------------------------- HTTP payloads
def test_chat_client_wire_format(tmp_path):
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append((str(request.url), request.headers["authorization"], body))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant",
                                                                  "content": "ok"}}]})

    c = ChatClient(base_url="http://fake:11434/v1/", model="m-tools", vlm_model="m-vision",
                   api_key="k")
    c.http = httpx.Client(transport=httpx.MockTransport(handler))

    assert c.chat([{"role": "user", "content": "hi"}], TOOLS)["content"] == "ok"
    url, auth, body = seen[0]
    assert url == "http://fake:11434/v1/chat/completions" and auth == "Bearer k"
    assert body["model"] == "m-tools" and len(body["tools"]) == len(TOOLS)

    img = tmp_path / "f.jpg"
    img.write_bytes(b"\xff\xd8abc")
    assert c.describe_image(img, "what?") == "ok"
    body = seen[1][2]
    assert body["model"] == "m-vision"
    parts = body["messages"][0]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
