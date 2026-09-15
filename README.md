# semubot_brain

ROS 2 (Jazzy) package for Semubot's conversational "brain": listens to the robot's
microphone, transcribes speech, generates a spoken-language response with a local
LLM, and speaks it back.

## What it does

Single node (`semu_brain_node`, script `llm_controller`) runs a loop:

1. **Listen** — subscribes to the `audio` topic (`audio_common_msgs/AudioData`,
   16-bit PCM, mono, 16kHz) and buffers incoming audio.
2. **Voice activity detection** — a simple RMS-energy threshold detects when
   someone is speaking, buffers audio while speech continues, and triggers
   processing after ~3 seconds of silence.
3. **Speech-to-text** — sends the buffered audio (wrapped as an in-memory WAV) to
   a Whisper server over HTTP.
4. **LLM response** — sends the transcribed text, plus a running chat history and
   a fixed Estonian-language system persona/prompt, to a llama.cpp server's
   `/completion` endpoint.
5. **Text-to-speech** — pipes the LLM's reply through Piper TTS and plays it with
   `aplay`.

Everything after step 2 runs in a background thread so the node keeps listening
without blocking.

## Requires external servers (not part of this package)

This node is a client — it does not host the STT/LLM models itself. Two HTTP
servers must already be running and reachable:

- **Whisper server** — expected at `http://192.168.0.172:9000/inference`
  (startup healthcheck currently pings `http://localhost:8081/`, which is a
  different host/port than the one actually used for transcription — worth
  fixing).
- **llama.cpp server** — expected at `http://192.168.0.172:8080/completion`.

Both endpoint URLs are hardcoded at the top of `llm_controller.py`
(`WHISPER_SERVER_URL`, `LLAMA_SERVER_URL`), as is the Piper voice model path
(`PIPER_VOICE_PATH`). Update those constants if server addresses or the voice
model change.

## Dependencies

- ROS 2 Jazzy, `rclpy`
- `audio_common_msgs` (used in code, but currently missing from `package.xml` —
  add it as a `<depend>`)
- Python: `requests`, `numpy`
- `piper` and `aplay` available on `PATH`/system for TTS playback

