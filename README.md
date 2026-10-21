# semubot_brain

ROS 2 (Jazzy) package for Semubot's conversational "brain". One node, `semu_brain_node`
(script `llm_controller`), owns the robot's **speech mode** and is the only process that speaks.

## Speech modes

| Mode | Mic -> LLM loop | Reply comes from |
|---|---|---|
| `manual` (default after start) | off | operator text / quick-chat from the webapp |
| `local` | on | llama.cpp server running TartuNLP Llama-3.1-EstLLM-8B-Instruct |
| `cloud` | on | Claude or OpenAI (`CLOUD_PROVIDER` in brain.env) |

Text published on `/speech_input` (webapp text form, quick chat) is spoken in **every** mode.
While the robot speaks or thinks, the mic is ignored so it never answers itself.
All replies are Estonian and spoken with the TartuNLP TTS (`~/tts-worker-test`, speaker `mari`).

## Topics

| Topic | Type | Direction |
|---|---|---|
| `/semubot/mode` | `std_msgs/String` (`manual`/`local`/`cloud`) | in (webapp button) |
| `/semubot/mode_state` | `std_msgs/String`, transient-local | out (current mode) |
| `/speech_input` | `std_msgs/String` | in (text to say) |
| `audio` | `audio_common_msgs/AudioData` (16 kHz mono int16) | in (audio_publisher) |
| `/semubot/audio_level` | `std_msgs/Float32MultiArray` `[rms, speech_prob]` | out (10 Hz) |
| `/semubot/vad_threshold` | `std_msgs/Float32` | in (loudness threshold) |
| `/semubot/vad_threshold_state` | `std_msgs/Float32`, transient-local | out |

The node runs on `ROS_DOMAIN_ID=5` with CycloneDDS (same as rosbridge/webapp); this is
set per node in `semubot_support/launch/semubot_bringup.launch.py`.

## Loop (local/cloud)

Speech detection -> whisper.cpp server -> LLM (last 6 exchanges of history) -> TTS -> `play`.
LLM failures are logged and the robot says "Vabandust, ma ei saa praegu mõelda.". Switching
mode clears the history; switching to `local` pre-warms llama.cpp's prompt cache (first uncached
prompt takes ~50 s on the NUC CPU, cached replies ~5-15 s).

## Speech detection (`semubot_brain/vad.py`)

Every 0.1 s audio chunk gets two numbers:

- **speech probability** from [Silero VAD](https://github.com/snakers4/silero-vad) (ONNX model,
  MIT licence, `~/.local/share/semubot/silero_vad.onnx`, ~0.3 ms CPU per chunk). It tells voice
  apart from noise (fans, music, bangs).
- **loudness** (RMS of the 16-bit samples).

A chunk counts as speech when probability >= 0.5 **and** loudness >= the threshold (default 1000).
The threshold rejects voices that are far from the robot. An utterance starts at the first speech
chunk (with 0.3 s of audio before it), ends after 1.2 s without speech or at 15 s (Whisper's limit
with `-ac 768`), and is dropped if it has less than 0.25 s of speech. Without the model or
onnxruntime, the node falls back to loudness only.

Every recording is logged (journal lines starting with `[REC]`, and `audio` entries in the
conversation log): when it starts (loudness and speech probability), and how it ends - sent to
Whisper (length, seconds of speech, ended by 1.2 s of silence or the 15 s limit) or discarded
(too little speech). If Whisper finds no words, that is logged too and nothing goes to the LLM.

The threshold is set on the chest display (admin menu -> Kõnerežiim -> Mikrofon) or
by publishing `std_msgs/Float32` on `/semubot/vad_threshold`. It is saved in
`~/.config/semubot/vad.json` and reported on `/semubot/vad_threshold_state` (latched). The live
level is published on `/semubot/audio_level` (`Float32MultiArray` `[rms, speech_probability]`,
10 times a second, in every mode).

## Voice

Voice and speed are set on the chest display (Kõnerežiim -> Hääl) or by publishing JSON on
`/semubot/tts_settings` (`std_msgs/String`, e.g. `{"speaker": "vesta", "speed": 1.2}`; either key
may be left out). Speakers are read from the TTS server at startup (`GET /v2`); speed is 0.5-2.0.
The current settings plus the available speakers are published latched on
`/semubot/tts_settings_state`. Settings are saved in `~/.config/semubot/tts.json`, which the
`speak` command also reads.

## Conversation history, log and cloud usage

- **History:** the last 6 exchanges are sent to the LLM. They are forgotten on a mode change and
  after **5 minutes without an exchange** (checked every 10 s), so the next visitor starts fresh.
- **Conversation log:** the last 30 events (`heard`, `said`, `typed`, `info`, `error`) with a
  timestamp, published as a JSON list (latched) on `/semubot/conversation_log` for the chest
  display. Kept in memory only, never written to disk; `std_msgs/Empty` on
  `/semubot/conversation_log_clear` empties it.
- **Cloud usage:** every cloud request (also refused ones) is counted per day in
  `~/.config/semubot/cloud_usage.json` (requests, input/output tokens, estimated USD cost; about
  two months kept). Today and this month are published as JSON (latched) on
  `/semubot/cloud_usage`. Prices for `claude-opus-5-5`, `claude-sonnet-5-5` and
  `claude-haiku-4-5` are built in (Anthropic list prices); for OpenAI or other models set
  `CLOUD_PRICE_INPUT` and `CLOUD_PRICE_OUTPUT` (USD per million tokens) in brain.env, otherwise
  only tokens are counted. The cost is an estimate; the provider's console is authoritative.

## Current stage

`/semubot/state` (`std_msgs/String`, JSON, latched, published on change) tells what the robot is
doing: `{"stage": ..., "mode": ..., "since": <unix time>}` with stage `manual` (not listening),
`listening`, `recording` (someone is talking), `transcribing` (Whisper), `thinking` (LLM),
`synthesizing` (TTS) or `speaking` (playback). Log entries for heard and said text carry a
`details` field with how long each step took. The chest display shows both on the Vestlus screen.

## Config: `~/.config/semubot/brain.env`

Read by the node at startup (process environment overrides it). Restart the bringup after edits.
Keys: `CLOUD_PROVIDER` (claude|openai), `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`
(default `claude-opus-5-5`), `OPENAI_API_KEY`, `OPENAI_MODEL`, `LOCAL_LLM_URL`, `WHISPER_URL`,
`TTS_URL`, `TTS_SPEAKER`, `VAD_MODEL`, `CLOUD_PRICE_INPUT`, `CLOUD_PRICE_OUTPUT`.

## Services it needs (systemd **user** services of `semubot`, lingering enabled)

- `llama-server.service` - EstLLM 8B Q4_K_M on `127.0.0.1:8082`
- `whisper-server.service` - TalTech Estonian whisper large-v3-turbo (`-t 8 -ac 768`, utterances up to ~15 s, ~8 s per transcription) on `127.0.0.1:8081`
- TTS API from `~/tts-worker-test` (docker compose) on `127.0.0.1:8001`

`systemctl --user status llama-server whisper-server`

## Dependencies

ROS 2 Jazzy (`rclpy`, `std_msgs`, `audio_common_msgs`), Python `requests`, `numpy`,
`anthropic`, `onnxruntime` and `brotli>=1.2.0` (installed with `pip install --user --break-system-packages anthropic onnxruntime "brotli>=1.2.0"`; Ubuntu's python3-brotli 1.1.0 is too old for the Anthropic SDK's HTTP client and makes every Claude request fail with a connection error), sox `play`.

## Tests

`cd src/semubot_brain && python3 -m pytest test/test_brain_logic.py test/test_vad.py test/test_tts_settings.py test/test_conversation_usage.py` (no ROS needed).
