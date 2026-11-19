# TaiwanSpeak Notes

**TaiwanSpeak Notes** is a local-first, privacy-focused real-time transcription, translation, and meeting summarization tool built for bilingual **Taiwan Mandarin–English** semiconductor engineering lectures, online lab syncs, and webinars.

Designed to capture audio directly from your computer's speaker output on lightweight laptops (such as an **Intel Core i5-1245U with 8 GB RAM**) as well as dedicated GPU workstations, it combines local `faster-whisper` speech recognition, domain-specific semiconductor terminology correction, rate-limit-resilient translation, and optional offline LLM summarization via **Ollama**.

## Key Features

* **Direct Speaker Audio Capture (WASAPI Loopback):** Records system playback audio directly from **Google Meet, Microsoft Teams, Zoom, or YouTube** without requiring virtual audio cables or risking room microphone noise.
* **Local-First Speech Recognition (`faster-whisper`):** Runs `small` (`int8` quantized) locally on CPU (or CUDA on NVIDIA GPUs) so sensitive research discussions, wafer lot data, and unpublished device results never leave your machine for speech-to-text.
* **Semiconductor Domain Adaptation:**
  * **Live-Reloaded `glossary.txt`:** Primes Whisper's decoder in real time with active session acronyms, cleanroom tools, and Taiwanese research institute names without restarting the app.
  * **Pre- & Post-Translation Term Normalizers:** Deterministically corrects common Taiwan Mandarin acoustic homophones (e.g., `直流電`, `二極體`, `功率`, `能隙`, `場板`, `磊晶`, `閎康`, `國網中心`) and technical units (`V`, `Id-Vg`, `VDMOS`, `C-DLTS`, `TCAD`).
* **Anti-Hallucination & Code-Switching Guards:** Uses Silero VAD filtering, a configurable Silence Gate, capped temperature fallback (`[0.0, 0.2]`), confidence gating (`avg_logprob`), and Unicode script filtering to prevent multilingual gibberish loops during silent pauses or cross-talk.
* **Rate-Limit-Resilient Translation (`HTTP 429` Protection):** Serializes outgoing translation requests with thread-safe pacing (`_rate_lock`) and automatic multi-endpoint failover (`gtx` → `dict-chrome-ex` → `MyMemory`) so long online meetings on shared campus/dormitory NAT IPs run uninterrupted.
* **Offline AI Meeting Summaries (`app_ollama_test.py`):** Integrates with local **Ollama** (`qwen2.5:3b`) to generate structured action items and technical summaries completely offline.
* **Interactive Web UI:** Displays timestamped English and Traditional Chinese (`zh-TW`) transcript cards, supports inline text editing, live topic hints, manual **Transcribe Now** buffer flushing, and one-click Markdown export.

## Project Structure

```text
TaiwanSpeak/
├── app.py                # Main speaker-source web app (WASAPI loopback capture + transcription + translation)
├── app_ollama_test.py    # Speaker-source build with integrated local Ollama (qwen2.5:3b) summarization
├── glossary.txt          # Dynamic semiconductor vocabulary bank (read live during capture)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Prerequisites

* **OS:** Windows 10 / 11 (required for native WASAPI loopback speaker capture)
* **Python:** Python 3.10 or 3.11
* **Hardware:**
  * *Minimum / Laptop:* Intel Core i5 (12th Gen or newer) + 8 GB RAM (`small` model with `int8` compute type)
  * *Recommended / Workstation:* Core i7 + 16–32 GB RAM + NVIDIA GPU (supports `medium` model / `float16`)
* **Optional (for Offline Summarization):** [Ollama](https://ollama.com/) installed locally with the `qwen2.5:3b` model pulled:
  ```bash
  ollama pull qwen2.5:3b
  ```

## Installation

1. **Open the project folder:**
   ```bash
   cd path/to/TaiwanSpeak
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Core packages include `fastapi`, `uvicorn`, `faster-whisper`, `soundcard` / `pyaudiowpatch`, `numpy`, and `silero-vad`.)*

## Usage

### 1. Start the Application

Ensure your Windows playback device (speakers or headphones) is active, then launch the speaker-capture server:

```bash
python app.py
```

Open your browser to the local UI address printed in the terminal (typically `http://127.0.0.1:8000`).

### 2. Recommended Runtime Settings

For fast-paced technical lectures and online Google Meet calls on an i5-1245U:

| Setting | Recommended Value | Why It Works Best |
| :--- | :--- | :--- |
| **Model Size** | `small` (`int8`) | Best balance of Taiwan Mandarin accuracy and ~1s inference speed on CPU. |
| **Chunk Slice** | `10 seconds` | Cuts Whisper encoder overhead in half vs. `5s`, avoids mid-word cuts on fast speakers, and halves translation API calls. |
| **Silence Gate** | `10%` (YouTube lecture) / `12%` (Google Meet) | Skips silent pauses between slides or speakers without clipping soft consonants. |
| **Show zh-TW** | **Checked** | Displays the original Traditional Chinese transcription beneath English cards so you can verify technical terms and phonetic homophones. |

### 3. Best Practices During Online Calls & Lectures

* **Start capture 10–15 seconds early:** Click **Start** slightly before the call or video begins so `faster-whisper` finishes loading into RAM before the opening introductions.
* **Use `Transcribe Now` for immediate updates:** Whenever a speaker finishes an important slide or point and you do not want to wait for the 10-second timer, click **Transcribe Now** to flush the audio buffer immediately.
* **Export Notes:** Click **Export Markdown** at the end of the session to download a clean `.md` transcript with timestamps.

## Customizing `glossary.txt`

`glossary.txt` is read dynamically during transcription—you can edit and save it mid-meeting without restarting the server.

* **Keep the top 10–15 lines session-specific:** Whisper enforces a 224-token limit on `initial_prompt`, so only the top ~450–750 characters of `glossary.txt` are active. Place the exact device architectures, tools, and collaborator names for **today's session** at the very top of the file.
* **Avoid unrelated acronyms in the active window:** Remove CMOS-only terms (such as `GAA` or `FinFET`) from the top lines during wide-bandgap power talks (`GaN`, `SiC`), as Whisper may force ambiguous syllables onto prompt keywords.
* **Use `#` for comments:** Lines starting with `#` are ignored automatically, allowing you to organize sections (e.g., `# Lithography`, `# Deposition`, `# Power Devices`) and copy the relevant block to the top before each session.

Example top block for a GaN Power Device sync:

```text
# === Active Session Terms ===
國立陽明交通大學, 台灣半導體研究中心, TSRI, 閎康, MA-tek, 國網中心
AlGaN/GaN, HEMT, VDMOS, TCAD, 3D TCAD, C-DLTS
能隙, 崩潰電壓, 導通電阻, 場板, 磊晶, 閘極漏電流, 鈍化層
Veeco ALD, PECVD, double-side aligner, spin coater, SEM, SIMS
```

## Troubleshooting

* **No audio captured from Google Meet or YouTube:**
  * Verify that Windows is outputting sound to your default playback device (WASAPI loopback captures from the active default output device) and that the browser tab is not muted.
* **`HTTP 429` (Too Many Requests) during long Google Meet calls:**
  * Ensure **Chunk Slice** is set to `10 seconds` (not `5 seconds`) so translation requests stay under 6 per minute.
  * The built-in `_rate_lock` spacer and automatic failover (`gtx` → `dict-chrome-ex` → `MyMemory`) handle shared campus IP limits automatically without stopping transcription.
* **Short embedded English acronyms (e.g., `"MIT"`, `"ALD"`) get skipped mid-sentence:**
  * Rapidly spoken 3-letter acronyms inside Mandarin sentences can blend into Chinese phonetic homophones (e.g., `愛提`) or get smoothed out by neural translators. Add the acronym to the top 10 lines of `glossary.txt` and keep **Show zh-TW** enabled to verify the raw Chinese capture.
* **First 5–10 seconds of a call are missed:**
  * On cold launch, loading `faster-whisper` (`small`) into RAM takes 2–4 seconds on CPU. Start capture a few seconds before the meeting begins, or ensure background model pre-warming is enabled in `lifespan()`.