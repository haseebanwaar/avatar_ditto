import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from PIL import Image

from ditto_api.app import create_app
from ditto_api.config import Settings


class FakeEngine:
    instances = []

    def __init__(self, settings):
        self.session = None
        self.gains = None
        self.instances.append(self)

    def setup(self, session):
        self.session = session

    def controls(self, motion, mouth, initial=False):
        self.gains = (motion, mouth)

    def run(self, chunk):
        assert chunk.shape == (6480,)
        for _ in range(5):
            self.session.on_frame(np.full((self.session.height, self.session.width, 3), 100, np.uint8))

    def close_session(self):
        pass


def portrait():
    b = io.BytesIO()
    Image.new("RGB", (300, 400), "#4a7a98").save(b, "PNG")
    return b.getvalue()


def speech():
    b = io.BytesIO()
    sf.write(b, np.sin(np.arange(16000) * 0.1).astype(np.float32) * 0.1, 16000, format="WAV")
    return b.getvalue()


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(Settings(checkpoints=tmp_path, runtime=tmp_path / "runtime"), FakeEngine)) as c:
        yield c


def create(client, **values):
    return client.post(
        "/v1/sessions", data={"resolution": "720", **values}, files={"avatar": ("avatar.png", portrait(), "image/png")}
    )


def test_resolution_controls_and_resident_reuse(client):
    r = create(client, motion_intensity=".4")
    assert r.status_code == 201, r.text
    d = r.json()
    assert (d["width"], d["height"], d["fps"]) == (540, 720, 25)
    assert create(client).status_code == 409
    assert client.patch(d["status_url"] + "/controls", json={"motion_intensity": 1.2}).status_code == 200
    assert FakeEngine.instances[-1].gains == (1.2, 0.65)
    assert client.delete(d["status_url"]).status_code == 200
    assert client.get(d["status_url"]).status_code == 404
    r = create(client, resolution="2048")
    assert r.status_code == 201 and r.json()["height"] == 2048
    assert client.get("/health").json()["model_load_count"] == 1
    client.delete(r.json()["status_url"])


@pytest.mark.parametrize(
    "values",
    [
        {"resolution": "999"},
        {"motion_intensity": "-.1"},
        {"motion_intensity": "nan"},
        {"motion_intensity": "2"},
        {"mouth_intensity": "inf"},
    ],
)
def test_invalid_parameters_do_not_claim_model(client, values):
    assert create(client, **values).status_code == 422
    assert client.get("/health").json()["active_sessions"] == 0


def test_bad_image_releases_lock(client):
    r = client.post("/v1/sessions", files={"avatar": ("bad.png", b"not an image", "image/png")})
    assert r.status_code == 422
    assert create(client).status_code == 201


def test_repeat_audio_and_websocket_end_preserve_session(client):
    d = create(client).json()
    for _ in range(2):
        r = client.post(d["audio_upload_url"], files={"audio": ("speech.wav", speech(), "audio/wav")})
        assert r.status_code == 202 and r.json()["accepted_seconds"] == 1
    with client.websocket_connect(d["audio_websocket"]) as ws:
        for _ in range(2):
            ws.send_bytes(np.zeros(1600, dtype="<i2").tobytes())
            ws.send_text(json.dumps({"type": "end"}))
            assert ws.receive_json() == {"type": "ack", "stream_continues": True}
    assert not client.get(d["status_url"]).json()["closed"]
    assert client.get("/health").json()["active_sessions"] == 1
    assert client.delete(d["status_url"]).status_code == 200


def test_no_offer_session_expires(client):
    import asyncio
    import time

    d = create(client).json()
    manager = client.app.state.manager
    manager.sessions[d["id"]].created = time.monotonic() - 100
    client.portal.call(asyncio.sleep, 5.1)
    assert client.get(d["status_url"]).status_code == 404
    assert create(client).status_code == 201


def test_bad_audio_does_not_break_session(client):
    d = create(client).json()
    r = client.post(d["audio_upload_url"], files={"audio": ("bad.wav", b"invalid", "audio/wav")})
    assert r.status_code == 422
    assert client.get(d["status_url"]).status_code == 200


def test_delete_closes_open_audio_socket(client):
    from starlette.websockets import WebSocketDisconnect

    d = create(client).json()
    with client.websocket_connect(d["audio_websocket"]) as ws:
        assert client.delete(d["status_url"]).status_code == 200
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()
    assert client.get("/health").json()["active_sessions"] == 0
