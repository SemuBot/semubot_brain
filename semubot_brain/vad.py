"""Speech detection: Silero VAD (is it a voice?) plus a loudness gate (is it close enough?)."""
import collections

import numpy as np

SAMPLE_RATE = 16000
FRAME = 512  # Silero frame at 16 kHz (32 ms)
CONTEXT = 64  # Samples of the previous frame Silero expects in front of each frame


class SileroVad:
    """Silero VAD ONNX model; returns the speech probability of 512-sample frames."""

    def __init__(self, model_path):
        import onnxruntime
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            model_path, sess_options=options, providers=['CPUExecutionProvider'])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, frame):
        """frame: FRAME float32 samples in [-1, 1]."""
        x = np.concatenate([self.context, frame.reshape(1, FRAME)], axis=1)
        out, self.state = self.session.run(None, {
            'input': x, 'state': self.state, 'sr': np.array(SAMPLE_RATE, dtype=np.int64)})
        self.context = x[:, -CONTEXT:]
        return float(out[0][0])


class SpeechDetector:
    """Cuts the microphone stream into utterances.

    A chunk counts as speech when the VAD speech probability is at least prob_threshold
    and its RMS loudness is at least `threshold` (the loudness gate rejects far-away
    voices). Time is measured from the audio itself, so behaviour is deterministic.
    """

    def __init__(self, vad, threshold=1000.0, prob_threshold=0.5, end_silence=1.2,
                 max_seconds=15.0, min_speech=0.25, preroll=0.3):
        self.vad = vad  # callable(frame) -> probability, or None for loudness only
        self.threshold = threshold
        self.prob_threshold = prob_threshold
        self.end_silence = end_silence
        self.max_seconds = max_seconds
        self.min_speech = min_speech
        self.preroll_seconds = preroll
        self.pending = np.zeros(0, dtype=np.float32)
        self.preroll = collections.deque()
        self.preroll_samples = 0
        self.last_event = None  # What the last process() call did, for logging (see process)
        self._clear()

    def _clear(self):
        self.recording = False
        self.buffer = []
        self.samples = 0
        self.speech_samples = 0
        self.silence_samples = 0

    def reset(self):
        """Drop any partial utterance (e.g. while the robot is talking)."""
        self._clear()
        self.preroll.clear()
        self.preroll_samples = 0

    def _speech_probability(self, audio):
        if self.vad is None:
            return 1.0
        samples = np.concatenate([self.pending, audio.astype(np.float32) / 32768.0])
        frames = len(samples) // FRAME
        self.pending = samples[frames * FRAME:]
        probs = [self.vad(samples[i * FRAME:(i + 1) * FRAME]) for i in range(frames)]
        return max(probs, default=0.0)

    def process(self, chunk, listening=True):
        """Feed one chunk of 16-bit mono PCM bytes.

        Returns (rms, speech_probability, utterance) where utterance is the raw PCM bytes
        of a finished utterance, or None. Sets last_event to None or a dict:
        {'event': 'started', 'rms', 'prob'} when a recording starts, or
        {'event': 'ended' | 'dropped', 'seconds', 'speech_seconds', 'reason': 'silence' |
        'max_length'} when it finishes ('dropped' = too little speech, nothing returned).
        """
        self.last_event = None
        audio = np.frombuffer(chunk, dtype=np.int16)
        if audio.size == 0:
            return 0.0, 0.0, None
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        prob = self._speech_probability(audio)
        if not listening:
            self.reset()
            return rms, prob, None

        is_speech = prob >= self.prob_threshold and rms >= self.threshold
        if not self.recording:
            self.preroll.append(chunk)
            self.preroll_samples += audio.size
            while self.preroll and self.preroll_samples - len(self.preroll[0]) // 2 >= \
                    self.preroll_seconds * SAMPLE_RATE:
                self.preroll_samples -= len(self.preroll.popleft()) // 2
            if not is_speech:
                return rms, prob, None
            self.recording = True
            self.buffer = list(self.preroll)
            self.samples = self.preroll_samples
            self.preroll.clear()
            self.preroll_samples = 0
            self.last_event = {'event': 'started', 'rms': rms, 'prob': prob}
        else:
            self.buffer.append(chunk)
            self.samples += audio.size

        if is_speech:
            self.speech_samples += audio.size
            self.silence_samples = 0
        else:
            self.silence_samples += audio.size

        ended = self.silence_samples >= self.end_silence * SAMPLE_RATE
        too_long = self.samples >= self.max_seconds * SAMPLE_RATE
        if not (ended or too_long):
            return rms, prob, None
        utterance = b''.join(self.buffer)
        enough_speech = self.speech_samples >= self.min_speech * SAMPLE_RATE
        self.last_event = {
            'event': 'ended' if enough_speech else 'dropped',
            'seconds': self.samples / SAMPLE_RATE,
            'speech_seconds': self.speech_samples / SAMPLE_RATE,
            'reason': 'silence' if ended else 'max_length',
        }
        self._clear()
        return rms, prob, utterance if enough_speech else None
