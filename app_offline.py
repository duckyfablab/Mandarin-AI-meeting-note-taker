import asyncio
import collections
import contextlib
import json
import os
import queue
import threading
import time

from datetime import datetime
from typing import Optional

import ctranslate2
import numpy as np
import pyaudiowpatch as pyaudio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel

# ==============================================================================
# Application State & Queues
# ==============================================================================

class AppState:
    def __init__(self):
        self.is_capturing: bool = False
        self.model: Optional[WhisperModel] = None
        self.loaded_model_size: Optional[str] = None
        self.model_lock = threading.Lock()

        self.raw_audio_queue: queue.Queue = queue.Queue()
        self.inference_queue: queue.Queue = queue.Queue()
        self.flush_event = threading.Event()

        self.chunk_counter: int = 0
        self.in_flight_jobs: int = 0
        self.segments: list[dict] = []

        # Live capture configuration (updated from UI)
        self.model_size: str = "small"
        self.chunk_seconds: int = 7
        self.vad_threshold: float = 12.0  # RMS % gate (0-100)
        self.topic_hint: str = ""
        self.include_zh: bool = False

        self.active_device_name: str = "Not started"
        self.connected_clients: set[WebSocket] = set()
        self.main_loop: Optional[asyncio.AbstractEventLoop] = None


state = AppState()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    state.main_loop = asyncio.get_running_loop()
    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()
    yield
    state.is_capturing = False


app = FastAPI(title="TaiwanSpeak Local Notes", lifespan=lifespan)


# ==============================================================================
# WebSocket Broadcast Helper
# ==============================================================================

async def _broadcast_async(payload: dict):
    if not state.connected_clients:
        return
    message = json.dumps(payload, ensure_ascii=False)
    stale = []
    for ws in list(state.connected_clients):
        try:
            await ws.send_text(message)
        except Exception:
            stale.append(ws)
    for ws in stale:
        state.connected_clients.discard(ws)


def broadcast(payload: dict):
    if state.main_loop and state.main_loop.is_running():
        asyncio.run_coroutine_threadsafe(_broadcast_async(payload), state.main_loop)


# ==============================================================================
# Whisper Model Loader (Auto-detects NVIDIA CUDA GPU or Multi-Core CPU)
# ==============================================================================

def get_whisper_model(model_size: str) -> WhisperModel:
    with state.model_lock:
        if state.model is not None and state.loaded_model_size == model_size:
            return state.model

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        broadcast({
            "type": "status",
            "status": "loading",
            "message": f"Loading Whisper '{model_size}' ({device.upper()}/{compute_type})..."
        })

        state.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=max(4, (os.cpu_count() or 4) // 2)
        )
        state.loaded_model_size = model_size
        return state.model


# ==============================================================================
# Audio Resampling & Windows WASAPI System Loopback Capture
# ==============================================================================

def resample_to_16k_mono(pcm_bytes: bytes, channels: int, src_rate: int) -> np.ndarray:
    """Convert raw int16 interleaved WASAPI frames into 16kHz mono float32 [-1, 1]."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        # Trim trailing partial frame if any, then average channels to mono
        usable = (len(samples) // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)

    if src_rate == 16000 or len(samples) == 0:
        return samples

    target_len = int(round(len(samples) * 16000.0 / src_rate))
    if target_len <= 1:
        return np.array([], dtype=np.float32)

    src_x = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(dst_x, src_x, samples).astype(np.float32)


def capture_loop_thread():
    """Captures Windows default playback output (speakers/headphones) via WASAPI loopback."""
    p = pyaudio.PyAudio()
    stream = None

    try:
        # Ensure Whisper model is downloaded and warm before streaming chunks
        get_whisper_model(state.model_size)

        loopback = p.get_default_wasapi_loopback()
        channels = int(loopback["maxInputChannels"])
        src_rate = int(loopback["defaultSampleRate"])
        state.active_device_name = loopback["name"]

        # Clear any stale audio
        while not state.raw_audio_queue.empty():
            try:
                state.raw_audio_queue.get_nowait()
            except queue.Empty:
                break

        def audio_callback(in_data, frame_count, time_info, status):
            if state.is_capturing and in_data:
                state.raw_audio_queue.put_nowait(in_data)
            return (None, pyaudio.paContinue)

        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=src_rate,
            input=True,
            input_device_index=loopback["index"],
            frames_per_buffer=2048,
            stream_callback=audio_callback,
        )
        stream.start_stream()

        broadcast({
            "type": "status",
            "status": "live",
            "device": state.active_device_name,
            "message": f"Capturing: {state.active_device_name}"
        })

        byte_chunks: list[bytes] = []
        buffered_bytes = 0
        peak_rms_percent = 0.0
        chunk_start_ts = datetime.now().strftime("%H:%M:%S")
        bytes_per_second = src_rate * channels * 2  # 16-bit = 2 bytes/sample

        while state.is_capturing:
            try:
                data = state.raw_audio_queue.get(timeout=0.2)
            except queue.Empty:
                # Windows WASAPI pushes zero packets when all apps are completely silent
                broadcast({"type": "rms", "rms": 0})
                if state.flush_event.is_set() and buffered_bytes > 0:
                    state.flush_event.clear()
                    _dispatch_buffer(
                        byte_chunks, channels, src_rate, peak_rms_percent, chunk_start_ts
                    )
                    byte_chunks = []
                    buffered_bytes = 0
                    peak_rms_percent = 0.0
                    chunk_start_ts = datetime.now().strftime("%H:%M:%S")
                continue

            if not byte_chunks:
                chunk_start_ts = datetime.now().strftime("%H:%M:%S")

            byte_chunks.append(data)
            buffered_bytes += len(data)

            # Calculate live RMS meter level
            frame_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(frame_np ** 2))) if len(frame_np) else 0.0
            rms_pct = min(100.0, round(rms * 350.0, 1))
            if rms_pct > peak_rms_percent:
                peak_rms_percent = rms_pct
            broadcast({"type": "rms", "rms": rms_pct})

            target_bytes = bytes_per_second * max(3, state.chunk_seconds)
            if buffered_bytes >= target_bytes or state.flush_event.is_set():
                state.flush_event.clear()
                _dispatch_buffer(
                    byte_chunks, channels, src_rate, peak_rms_percent, chunk_start_ts
                )
                byte_chunks = []
                buffered_bytes = 0
                peak_rms_percent = 0.0

        # Flush any remaining audio when user stops capture
        if buffered_bytes > bytes_per_second * 0.6:
            _dispatch_buffer(
                byte_chunks, channels, src_rate, peak_rms_percent, chunk_start_ts
            )

    except Exception as exc:
        state.is_capturing = False
        broadcast({"type": "error", "message": f"WASAPI Capture Error: {exc}"})
    finally:
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.stop_stream()
                stream.close()
        with contextlib.suppress(Exception):
            p.terminate()

        broadcast({"type": "rms", "rms": 0})
        broadcast({
            "type": "status",
            "status": "idle",
            "message": "Stopped"
        })


def _dispatch_buffer(
    byte_chunks: list[bytes],
    channels: int,
    src_rate: int,
    peak_rms_percent: float,
    timestamp: str,
):
    if not byte_chunks:
        return
    if peak_rms_percent < state.vad_threshold:
        return  # Below silence gate

    raw_bytes = b"".join(byte_chunks)
    audio_16k = resample_to_16k_mono(raw_bytes, channels, src_rate)
    if len(audio_16k) < 16000 * 0.5:
        return

    state.chunk_counter += 1
    state.inference_queue.put((state.chunk_counter, timestamp, audio_16k))


# ==============================================================================
# Online Whisper Translation & Transcription Worker
# ==============================================================================

def inference_worker():
    """Processes buffered 16kHz chunks into English translations (+ optional zh-TW)."""
    while True:
        chunk_id, timestamp, audio_16k = state.inference_queue.get()
        state.in_flight_jobs += 1
        broadcast({"type": "queue", "count": state.in_flight_jobs})

        try:
            model = get_whisper_model(state.model_size)

            # Build initial prompt to prime domain vocabulary (including English acronyms)
            prompt_parts = ["Taiwan Mandarin engineering and business meeting."]
            if state.topic_hint.strip():
                prompt_parts.append(f"Vocabulary: {state.topic_hint.strip()}.")
            # Include tail of previous English sentence for cross-chunk continuity
            if state.segments:
                prompt_parts.append(state.segments[-1]["en"][-140:])
            en_prompt = " ".join(prompt_parts)

            # 1. Direct Mandarin -> English speech translation with built-in Silero VAD
            segments_en, _ = model.transcribe(
                audio_16k,
                language="zh",
                task="translate",
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 450},
                condition_on_previous_text=False,
                initial_prompt=en_prompt,
            )
            english_text = " ".join(seg.text.strip() for seg in segments_en).strip()

            if english_text:
                zh_text = ""
                # 2. Optional Traditional Chinese (zh-TW) verbatim pass
                if state.include_zh:
                    zh_prompt = "以下是台灣的繁體中文會議紀錄，保留英文技術名詞。"
                    if state.topic_hint.strip():
                        zh_prompt += f" 關鍵字：{state.topic_hint.strip()}"
                    segments_zh, _ = model.transcribe(
                        audio_16k,
                        language="zh",
                        task="transcribe",
                        beam_size=3,
                        vad_filter=True,
                        condition_on_previous_text=False,
                        initial_prompt=zh_prompt,
                    )
                    zh_text = "".join(seg.text.strip() for seg in segments_zh).strip()

                seg_obj = {
                    "id": f"seg-{chunk_id}-{int(time.time() * 1000)}",
                    "order": chunk_id,
                    "time": timestamp,
                    "en": english_text,
                    "zh": zh_text,
                }
                state.segments.append(seg_obj)
                state.segments.sort(key=lambda x: x["order"])
                broadcast({"type": "segment", "segment": seg_obj})

        except Exception as exc:
            broadcast({"type": "error", "message": f"Whisper Inference Error: {exc}"})
        finally:
            state.in_flight_jobs = max(0, state.in_flight_jobs - 1)
            broadcast({"type": "queue", "count": state.in_flight_jobs})
            state.inference_queue.task_done()

*
# ==============================================================================
# FastAPI Routes & WebSocket Endpoint
# ==============================================================================

class StartConfig(BaseModel):
    model_size: str = "small"
    chunk_seconds: int = 7
    vad_threshold: float = 12.0
    topic_hint: str = ""
    include_zh: bool = False


class UpdateConfig(BaseModel):
    chunk_seconds: Optional[int] = None
    vad_threshold: Optional[float] = None
    topic_hint: Optional[str] = None
    include_zh: Optional[bool] = None


class EditSegmentBody(BaseModel):
    en: str


@app.post("/api/start")
async def api_start(cfg: StartConfig):
    if state.is_capturing:
        return {"ok": True, "message": "Already capturing"}

    state.model_size = cfg.model_size
    state.chunk_seconds = cfg.chunk_seconds
    state.vad_threshold = cfg.vad_threshold
    state.topic_hint = cfg.topic_hint
    state.include_zh = cfg.include_zh
    state.is_capturing = True

    t = threading.Thread(target=capture_loop_thread, daemon=True)
    t.start()
    return {"ok": True}


@app.post("/api/stop")
async def api_stop():
    state.is_capturing = False
    return {"ok": True}


@app.post("/api/flush")
async def api_flush():
    if state.is_capturing:
        state.flush_event.set()
    return {"ok": True}


@app.post("/api/config")
async def api_update_config(cfg: UpdateConfig):
    if cfg.chunk_seconds is not None:
        state.chunk_seconds = cfg.chunk_seconds
    if cfg.vad_threshold is not None:
        state.vad_threshold = cfg.vad_threshold
    if cfg.topic_hint is not None:
        state.topic_hint = cfg.topic_hint
    if cfg.include_zh is not None:
        state.include_zh = cfg.include_zh
    return {"ok": True}


@app.post("/api/clear")
async def api_clear():
    state.segments.clear()
    state.chunk_counter = 0
    return {"ok": True}


@app.patch("/api/segments/{seg_id}")
async def api_edit_segment(seg_id: str, body: EditSegmentBody):
    for seg in state.segments:
        if seg["id"] == seg_id:
            seg["en"] = body.en.strip() or seg["en"]
            return {"ok": True, "segment": seg}
    return {"ok": False}


@app.delete("/api/segments/{seg_id}")
async def api_delete_segment(seg_id: str):
    state.segments = [s for s in state.segments if s["id"] != seg_id]
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.connected_clients.add(ws)
    # Sync initial state upon connect/refresh
    await ws.send_text(json.dumps({
        "type": "init",
        "is_capturing": state.is_capturing,
        "device": state.active_device_name,
        "segments": state.segments,
    }, ensure_ascii=False))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        state.connected_clients.discard(ws)


# ==============================================================================
# Embedded English Web UI
# ==============================================================================

HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TaiwanSpeak Local — Offline WASAPI Meeting Translator</title>
  <style>
    :root {
      --bg-app: #0b0f19;
      --bg-panel: #111827;
      --bg-card: #161f32;
      --bg-input: #0d1322;
      --border: #243049;
      --border-focus: #6366f1;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --emerald: #10b981;
      --emerald-bg: rgba(16, 185, 129, 0.12);
      --danger: #ef4444;
      --danger-bg: rgba(239, 68, 68, 0.12);
      --amber: #f59e0b;
      --radius: 12px;
      --radius-sm: 8px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
      background: var(--bg-app);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: var(--bg-panel);
      border-bottom: 1px solid var(--border);
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-icon {
      width: 38px; height: 38px; border-radius: 10px;
      background: linear-gradient(135deg, #10b981, #6366f1);
      display: flex; align-items: center; justify-content: center;
      font-weight: 800; color: #fff; font-size: 0.95rem;
    }
    .brand h1 { font-size: 1.08rem; font-weight: 700; display: flex; align-items: center; gap: 8px; }
    .badge {
      font-size: 0.7rem; font-weight: 600; padding: 2px 8px; border-radius: 999px;
      background: rgba(16, 185, 129, 0.15); color: #6ee7b7;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .brand p { font-size: 0.78rem; color: var(--text-muted); }
    .device-chip {
      font-size: 0.78rem; color: var(--text-muted);
      background: var(--bg-input); border: 1px solid var(--border);
      padding: 6px 12px; border-radius: 999px;
    }
    .device-chip strong { color: var(--text-main); }

    #alertBanner {
      display: none; margin: 14px 24px 0; padding: 11px 16px;
      border-radius: var(--radius-sm); background: var(--danger-bg);
      border: 1px solid rgba(239, 68, 68, 0.35); color: #fca5a5;
      font-size: 0.84rem; align-items: center; justify-content: space-between;
    }

    .control-deck {
      margin: 16px 24px 0; background: var(--bg-panel);
      border: 1px solid var(--border); border-radius: var(--radius);
      padding: 16px 20px; display: flex; flex-direction: column; gap: 14px;
    }
    .deck-row {
      display: flex; align-items: center; justify-content: space-between;
      gap: 16px; flex-wrap: wrap;
    }
    .btn {
      display: inline-flex; align-items: center; justify-content: center; gap: 7px;
      height: 36px; padding: 0 14px; border-radius: var(--radius-sm);
      font-size: 0.83rem; font-weight: 600; cursor: pointer;
      border: 1px solid transparent; transition: all 0.15s ease;
    }
    .btn:disabled { opacity: 0.45; cursor: not-allowed; }
    .btn-emerald { background: var(--emerald); color: #052e16; }
    .btn-emerald:hover:not(:disabled) { background: #059669; color: #fff; }
    .btn-danger { background: var(--danger); color: #fff; }
    .btn-danger:hover:not(:disabled) { background: #dc2626; }
    .btn-primary { background: var(--accent); color: #fff; }
    .btn-primary:hover:not(:disabled) { background: var(--accent-hover); }
    .btn-secondary { background: var(--bg-card); color: var(--text-main); border-color: var(--border); }
    .btn-secondary:hover:not(:disabled) { background: #1e293b; }

    .status-pill {
      display: inline-flex; align-items: center; gap: 7px;
      padding: 5px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
      background: var(--bg-card); border: 1px solid var(--border); color: var(--text-muted);
    }
    .status-pill.live {
      background: var(--emerald-bg); border-color: rgba(16, 185, 129, 0.35); color: #34d399;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-dim); }
    .status-pill.live .dot { background: var(--emerald); box-shadow: 0 0 8px var(--emerald); }

    .meter-box {
      display: flex; align-items: center; gap: 12px;
      background: var(--bg-input); border: 1px solid var(--border);
      padding: 7px 14px; border-radius: var(--radius-sm); min-width: 240px;
    }
    .rms-track {
      flex: 1; height: 7px; background: #1e293b; border-radius: 999px;
      overflow: hidden; position: relative;
    }
    .rms-fill {
      height: 100%; width: 0%;
      background: linear-gradient(90deg, #10b981, #6366f1);
      transition: width 0.08s linear;
    }
    .vad-marker { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--amber); left: 12%; }

    .settings-grid {
      display: flex; align-items: flex-end; gap: 16px; flex-wrap: wrap;
      padding-top: 10px; border-top: 1px solid rgba(36, 48, 73, 0.7);
    }
    .field { display: flex; flex-direction: column; gap: 4px; font-size: 0.74rem; color: var(--text-muted); }
    .field select, .field input[type="text"] {
      background: var(--bg-input); border: 1px solid var(--border);
      color: var(--text-main); border-radius: 6px; padding: 6px 10px;
      font-size: 0.81rem; outline: none;
    }
    .field select:focus, .field input[type="text"]:focus { border-color: var(--border-focus); }
    .checkbox-label {
      display: inline-flex; align-items: center; gap: 7px;
      font-size: 0.81rem; color: var(--text-main); cursor: pointer; height: 32px;
    }
    .checkbox-label input { accent-color: var(--emerald); width: 15px; height: 15px; }

    main {
      flex: 1; padding: 18px 24px 24px; display: flex; flex-direction: column;
    }
    .panel {
      flex: 1; background: var(--bg-panel); border: 1px solid var(--border);
      border-radius: var(--radius); display: flex; flex-direction: column;
      min-height: 460px; max-height: calc(100vh - 250px);
    }
    .panel-header {
      padding: 13px 18px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between;
      gap: 10px; flex-wrap: wrap;
    }
    .panel-title { font-size: 0.94rem; font-weight: 700; display: flex; align-items: center; gap: 8px; }
    .toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .search-box {
      background: var(--bg-input); border: 1px solid var(--border);
      color: var(--text-main); border-radius: 6px; padding: 5px 10px;
      font-size: 0.8rem; width: 190px; outline: none;
    }
    .panel-body {
      flex: 1; overflow-y: auto; padding: 16px 18px;
      display: flex; flex-direction: column; gap: 11px;
    }
    .empty-state {
      margin: auto; text-align: center; max-width: 440px; color: var(--text-muted);
    }
    .empty-state h3 { color: var(--text-main); font-size: 1rem; margin-bottom: 8px; }
    .empty-state p { font-size: 0.84rem; line-height: 1.6; }

    .segment-card {
      background: var(--bg-card); border: 1px solid var(--border);
      border-radius: var(--radius-sm); padding: 12px 15px;
    }
    .segment-card:hover { border-color: #364769; }
    .segment-meta {
      display: flex; align-items: center; justify-content: space-between;
      font-size: 0.74rem; color: var(--text-dim); margin-bottom: 5px;
    }
    .icon-btn {
      background: transparent; border: none; color: var(--text-muted);
      cursor: pointer; padding: 2px 6px; border-radius: 4px; font-size: 0.82rem;
    }
    .icon-btn:hover { color: var(--text-main); background: rgba(255,255,255,0.07); }
    .segment-en { font-size: 0.96rem; font-weight: 600; color: var(--text-main); line-height: 1.5; }
    .segment-zh {
      margin-top: 6px; padding-top: 6px;
      border-top: 1px dashed rgba(148, 163, 184, 0.2);
      font-size: 0.83rem; color: var(--text-muted);
    }
    .edit-box {
      width: 100%; background: var(--bg-input); border: 1px solid var(--border-focus);
      color: var(--text-main); border-radius: 6px; padding: 8px;
      font-family: inherit; font-size: 0.9rem; min-height: 56px;
    }
    .queue-pill {
      display: none; align-items: center; gap: 6px; font-size: 0.76rem;
      color: #a5b4fc; background: rgba(99, 102, 241, 0.14);
      padding: 4px 10px; border-radius: 999px;
    }
    .spinner {
      width: 11px; height: 11px; border: 2px solid rgba(165,180,252,0.3);
      border-top-color: #a5b4fc; border-radius: 50%; animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-icon">TW</div>
      <div>
        <h1>TaiwanSpeak Local <span class="badge">100% Offline • WASAPI Loopback</span></h1>
        <p>Direct Windows speaker/headphone capture → Local Whisper English translation</p>
      </div>
    </div>
    <div class="device-chip">Output Device: <strong id="deviceName">Ready (Auto-detects default speakers/headphones)</strong></div>
  </header>

  <div id="alertBanner" role="alert">
    <span id="alertText"></span>
    <button class="icon-btn" onclick="document.getElementById('alertBanner').style.display='none'">✕</button>
  </div>

  <section class="control-deck">
    <div class="deck-row">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <button class="btn btn-emerald" id="startBtn">▶ Start System Capture</button>
        <button class="btn btn-danger" id="stopBtn" disabled>■ Stop</button>
        <button class="btn btn-secondary" id="flushBtn" disabled>Transcribe Now</button>
        <div class="status-pill" id="statusPill">
          <span class="dot"></span>
          <span id="statusText">Idle</span>
        </div>
        <div class="queue-pill" id="queuePill">
          <span class="spinner"></span>
          <span id="queueText">Translating...</span>
        </div>
      </div>

      <div class="meter-box">
        <span style="font-size:0.74rem;color:var(--text-muted);">System Audio</span>
        <div class="rms-track">
          <div class="rms-fill" id="rmsFill"></div>
          <div class="vad-marker" id="vadMarker"></div>
        </div>
        <span id="rmsPct" style="font-size:0.74rem;color:var(--text-muted);width:36px;text-align:right;">0%</span>
      </div>
    </div>

    <div class="settings-grid">
      <div class="field">
        <label for="modelSize">Whisper Model</label>
        <select id="modelSize">
          <option value="base">base (~150 MB • Fastest CPU)</option>
          <option value="small" selected>small (~460 MB • Recommended)</option>
          <option value="medium">medium (~1.5 GB • Best Technical Jargon)</option>
        </select>
      </div>

      <div class="field">
        <label for="chunkSeconds">Chunk Slice</label>
        <select id="chunkSeconds">
          <option value="5">5 seconds</option>
          <option value="7" selected>7 seconds (Balanced)</option>
          <option value="10">10 seconds</option>
        </select>
      </div>

      <div class="field">
        <label for="vadSlider">Silence Gate: <strong id="vadVal">12%</strong></label>
        <input type="range" id="vadSlider" min="0" max="45" value="12" style="accent-color:var(--amber);width:110px;height:30px;" />
      </div>

      <div class="field">
        <label for="topicHint">Domain / Technical Terms Hint</label>
        <input type="text" id="topicHint" placeholder="e.g., semiconductor, epitaxy, GaN, wafer" style="width:240px;" />
      </div>

      <label class="checkbox-label" title="Runs a second pass to also display Traditional Chinese text">
        <input type="checkbox" id="includeZh" />
        Also transcribe zh-TW subtitles
      </label>
    </div>
  </section>

  <main>
    <section class="panel">
      <div class="panel-header">
        <div class="panel-title">
          Live English Meeting Notes
          <span class="badge" id="segCount">0 segments</span>
        </div>
        <div class="toolbar">
          <input type="text" class="search-box" id="searchBox" placeholder="Search English or zh-TW..." />
          <button class="btn btn-secondary" id="copyAllBtn" style="height:30px;padding:0 10px;font-size:0.78rem;">Copy All</button>
          <button class="btn btn-secondary" id="exportMdBtn" style="height:30px;padding:0 10px;font-size:0.78rem;">Export .md</button>
          <button class="btn btn-secondary" id="clearBtn" style="height:30px;padding:0 10px;font-size:0.78rem;color:#fca5a5;">Clear</button>
        </div>
      </div>

      <div class="panel-body" id="transcriptList">
        <div class="empty-state">
          <h3>Ready to capture Windows speaker/headphone audio</h3>
          <p>
            Click <strong>Start System Capture</strong> and join your Teams, Zoom, LINE, or browser meeting.<br/><br/>
            Audio is tapped directly from your Windows playback device—works silently with headphones and ignores your microphone.
          </p>
        </div>
      </div>
    </section>
  </main>

  <script>
    let segments = [];
    let searchQuery = "";

    const el = {
      startBtn: document.getElementById("startBtn"),
      stopBtn: document.getElementById("stopBtn"),
      flushBtn: document.getElementById("flushBtn"),
      statusPill: document.getElementById("statusPill"),
      statusText: document.getElementById("statusText"),
      deviceName: document.getElementById("deviceName"),
      queuePill: document.getElementById("queuePill"),
      queueText: document.getElementById("queueText"),
      rmsFill: document.getElementById("rmsFill"),
      rmsPct: document.getElementById("rmsPct"),
      vadSlider: document.getElementById("vadSlider"),
      vadVal: document.getElementById("vadVal"),
      vadMarker: document.getElementById("vadMarker"),
      modelSize: document.getElementById("modelSize"),
      chunkSeconds: document.getElementById("chunkSeconds"),
      topicHint: document.getElementById("topicHint"),
      includeZh: document.getElementById("includeZh"),
      segCount: document.getElementById("segCount"),
      searchBox: document.getElementById("searchBox"),
      transcriptList: document.getElementById("transcriptList"),
      copyAllBtn: document.getElementById("copyAllBtn"),
      exportMdBtn: document.getElementById("exportMdBtn"),
      clearBtn: document.getElementById("clearBtn"),
      alertBanner: document.getElementById("alertBanner"),
      alertText: document.getElementById("alertText"),
    };

    function showAlert(msg) {
      el.alertText.textContent = msg;
      el.alertBanner.style.display = "flex";
    }

    function escapeHtml(str) {
      const d = document.createElement("div");
      d.textContent = str;
      return d.innerHTML;
    }

    function setCapturingUI(live) {
      el.startBtn.disabled = live;
      el.modelSize.disabled = live;
      el.stopBtn.disabled = !live;
      el.flushBtn.disabled = !live;
      if (live) {
        el.statusPill.classList.add("live");
      } else {
        el.statusPill.classList.remove("live");
      }
    }

    // WebSocket connection to local FastAPI server
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/ws`);

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "init") {
        segments = msg.segments || [];
        if (msg.device) el.deviceName.textContent = msg.device;
        setCapturingUI(msg.is_capturing);
        el.statusText.textContent = msg.is_capturing ? "Capturing" : "Idle";
        renderSegments();
      } else if (msg.type === "status") {
        el.statusText.textContent = msg.message;
        if (msg.device) el.deviceName.textContent = msg.device;
        setCapturingUI(msg.status === "live" || msg.status === "loading");
      } else if (msg.type === "rms") {
        el.rmsFill.style.width = `${msg.rms}%`;
        el.rmsPct.textContent = `${Math.round(msg.rms)}%`;
      } else if (msg.type === "queue") {
        if (msg.count > 0) {
          el.queuePill.style.display = "inline-flex";
          el.queueText.textContent = msg.count === 1 ? "Translating chunk..." : `Translating ${msg.count} chunks...`;
        } else {
          el.queuePill.style.display = "none";
        }
      } else if (msg.type === "segment") {
        segments.push(msg.segment);
        segments.sort((a, b) => a.order - b.order);
        renderSegments();
      } else if (msg.type === "error") {
        showAlert(msg.message);
      }
    };

    function renderSegments() {
      el.segCount.textContent = `${segments.length} segment${segments.length === 1 ? "" : "s"}`;
      const q = searchQuery.toLowerCase();
      const filtered = segments.filter(
        (s) => !q || s.en.toLowerCase().includes(q) || (s.zh && s.zh.toLowerCase().includes(q))
      );

      if (segments.length === 0) {
        el.transcriptList.innerHTML = `
          <div class="empty-state">
            <h3>Ready to capture Windows speaker/headphone audio</h3>
            <p>
              Click <strong>Start System Capture</strong> and join your Teams, Zoom, LINE, or browser meeting.<br/><br/>
              Audio is tapped directly from your Windows playback device—works silently with headphones and ignores your microphone.
            </p>
          </div>`;
        return;
      }

      const atBottom =
        el.transcriptList.scrollHeight - el.transcriptList.scrollTop - el.transcriptList.clientHeight < 60;

      el.transcriptList.innerHTML = filtered.map((s) => `
        <article class="segment-card">
          <div class="segment-meta">
            <span>#${s.order} • ${escapeHtml(s.time)}</span>
            <div>
              <button class="icon-btn" onclick="editSeg('${s.id}')" title="Edit">✎</button>
              <button class="icon-btn" onclick="deleteSeg('${s.id}')" title="Delete">✕</button>
            </div>
          </div>
          <div class="segment-en" id="en-${s.id}">${escapeHtml(s.en)}</div>
          ${s.zh ? `<div class="segment-zh">${escapeHtml(s.zh)}</div>` : ""}
        </article>
      `).join("");

      if (atBottom) {
        el.transcriptList.scrollTop = el.transcriptList.scrollHeight;
      }
    }

    async function deleteSeg(id) {
      await fetch(`/api/segments/${id}`, { method: "DELETE" });
      segments = segments.filter((s) => s.id !== id);
      renderSegments();
    }

    function editSeg(id) {
      const seg = segments.find((s) => s.id === id);
      const box = document.getElementById(`en-${id}`);
      if (!seg || !box || box.querySelector("textarea")) return;

      box.innerHTML = `
        <textarea class="edit-box">${escapeHtml(seg.en)}</textarea>
        <div style="display:flex;gap:6px;margin-top:6px;">
          <button class="btn btn-primary" id="save-${id}" style="height:26px;padding:0 10px;font-size:0.75rem;">Save</button>
          <button class="btn btn-secondary" onclick="renderSegments()" style="height:26px;padding:0 10px;font-size:0.75rem;">Cancel</button>
        </div>`;
      const ta = box.querySelector("textarea");
      ta.focus();
      document.getElementById(`save-${id}`).onclick = async () => {
        const updated = ta.value.trim();
        if (updated) {
          await fetch(`/api/segments/${id}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ en: updated }),
          });
          seg.en = updated;
        }
        renderSegments();
      };
    }

    // Live config sync
    async function syncLiveConfig() {
      await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chunk_seconds: Number(el.chunkSeconds.value),
          vad_threshold: Number(el.vadSlider.value),
          topic_hint: el.topicHint.value,
          include_zh: el.includeZh.checked,
        }),
      });
    }

    el.vadSlider.addEventListener("input", () => {
      el.vadVal.textContent = `${el.vadSlider.value}%`;
      el.vadMarker.style.left = `${el.vadSlider.value}%`;
      syncLiveConfig();
    });
    el.chunkSeconds.addEventListener("change", syncLiveConfig);
    el.topicHint.addEventListener("change", syncLiveConfig);
    el.includeZh.addEventListener("change", syncLiveConfig);

    el.startBtn.addEventListener("click", async () => {
      el.alertBanner.style.display = "none";
      setCapturingUI(true);
      el.statusText.textContent = "Initializing...";
      await fetch("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_size: el.modelSize.value,
          chunk_seconds: Number(el.chunkSeconds.value),
          vad_threshold: Number(el.vadSlider.value),
          topic_hint: el.topicHint.value,
          include_zh: el.includeZh.checked,
        }),
      });
    });

    el.stopBtn.addEventListener("click", () => fetch("/api/stop", { method: "POST" }));
    el.flushBtn.addEventListener("click", () => fetch("/api/flush", { method: "POST" }));

    el.searchBox.addEventListener("input", (e) => {
      searchQuery = e.target.value;
      renderSegments();
    });

    el.clearBtn.addEventListener("click", async () => {
      await fetch("/api/clear", { method: "POST" });
      segments = [];
      renderSegments();
    });

    function formatMarkdown() {
      const dateStr = new Date().toLocaleDateString();
      return [
        `# Meeting Transcript (${dateStr})`,
        ``,
        ...segments.map((s) => `**[${s.time}]** ${s.en}${s.zh ? `\n> ${s.zh}` : ""}`)
      ].join("\n\n");
    }

    el.copyAllBtn.addEventListener("click", async () => {
      if (!segments.length) return;
      await navigator.clipboard.writeText(formatMarkdown());
      el.copyAllBtn.textContent = "Copied ✓";
      setTimeout(() => (el.copyAllBtn.textContent = "Copy All"), 1400);
    });

    el.exportMdBtn.addEventListener("click", () => {
      if (!segments.length) return;
      const blob = new Blob([formatMarkdown()], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `meeting-notes-${Date.now()}.md`;
      a.click();
      URL.revokeObjectURL(url);
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_UI


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)