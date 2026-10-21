import pytest
import numpy as np

from semubot_brain.vad import FRAME, SpeechDetector

CHUNK = 1600  # 0.1 s, as published by audio_publisher_node


def chunk(level):
    """0.1 s of audio whose RMS is `level` (a square wave)."""
    return (np.where(np.arange(CHUNK) % 2, level, -level)).astype(np.int16).tobytes()


class FakeVad:
    """Says 'speech' for frames whose amplitude is in voice_levels."""

    def __init__(self, voice_levels):
        self.voice_levels = voice_levels
        self.frames = 0

    def __call__(self, frame):
        self.frames += 1
        assert frame.shape == (FRAME,)
        return 0.9 if round(abs(frame[0]) * 32768) in self.voice_levels else 0.1


def feed(detector, levels, listening=True):
    return [detector.process(chunk(level), listening) for level in levels]


def test_utterance_ends_after_silence_and_includes_preroll():
    det = SpeechDetector(FakeVad({3000}), threshold=1000, end_silence=0.5, preroll=0.3)
    results = feed(det, [100] * 5 + [3000] * 10 + [100] * 4)
    assert all(r[2] is None for r in results)
    assert det.recording
    rms, prob, utterance = det.process(chunk(100))
    assert prob < 0.5 and abs(rms - 100) < 1
    # 3 preroll chunks (incl. the first speech chunk) + 9 more speech + 5 silence
    assert len(utterance) == (3 + 9 + 5) * CHUNK * 2
    assert not det.recording


def test_loud_noise_is_not_speech():
    det = SpeechDetector(FakeVad({3000}), threshold=1000)
    results = feed(det, [20000] * 30)  # loud, but the VAD says it's not a voice
    assert not det.recording and all(r[2] is None for r in results)


def test_quiet_voice_below_threshold_is_ignored():
    det = SpeechDetector(FakeVad({500}), threshold=1000)
    feed(det, [500] * 30)  # a voice, but too far away
    assert not det.recording
    det.threshold = 400
    feed(det, [500])
    assert det.recording


def test_max_length_cuts_utterance():
    det = SpeechDetector(FakeVad({3000}), threshold=1000, max_seconds=2.0, preroll=0.1)
    results = feed(det, [3000] * 25)
    cut = [r[2] for r in results if r[2] is not None]
    assert len(cut) == 1 and len(cut[0]) == 20 * CHUNK * 2


def test_short_click_is_dropped():
    det = SpeechDetector(FakeVad({3000}), threshold=1000, end_silence=0.3, min_speech=0.25)
    results = feed(det, [3000, 3000] + [100] * 5)  # 0.2 s of speech < 0.25 s
    assert all(r[2] is None for r in results) and not det.recording


def test_not_listening_reports_levels_but_never_records():
    det = SpeechDetector(FakeVad({3000}), threshold=1000)
    rms, prob, utterance = det.process(chunk(3000), listening=False)
    assert abs(rms - 3000) < 1 and prob > 0.5 and utterance is None
    assert not det.recording


def test_frames_carry_over_between_chunks():
    vad = FakeVad({3000})
    det = SpeechDetector(vad)
    feed(det, [3000] * 16)  # 16 * 1600 = 25600 samples = 50 frames exactly
    assert vad.frames == 50


def test_loudness_only_without_vad():
    det = SpeechDetector(None, threshold=1000, end_silence=0.3)
    results = feed(det, [3000] * 5 + [100] * 3)
    assert results[-1][2] is not None


def test_events_describe_each_recording():
    det = SpeechDetector(FakeVad({3000}), threshold=1000, end_silence=0.3, min_speech=0.25,
                         preroll=0.1, max_seconds=2.0)
    events = []
    for level in [100, 3000, 3000, 3000] + [100] * 3 + [3000, 3000] + [100] * 3 + [3000] * 20:
        det.process(chunk(level))
        if det.last_event:
            events.append(det.last_event)
    assert [e['event'] for e in events] == ['started', 'ended', 'started', 'dropped',
                                            'started', 'ended']
    assert events[0]['rms'] == pytest.approx(3000) and events[0]['prob'] > 0.5
    assert events[1]['reason'] == 'silence' and events[1]['speech_seconds'] == pytest.approx(0.3)
    assert events[1]['seconds'] == pytest.approx(0.6)  # 0.3 s speech + 0.3 s silence
    assert events[3]['reason'] == 'silence' and events[3]['speech_seconds'] == pytest.approx(0.2)
    assert events[5]['reason'] == 'max_length' and events[5]['seconds'] == pytest.approx(2.0)


def test_no_event_while_nothing_happens():
    det = SpeechDetector(FakeVad({3000}), threshold=1000)
    det.process(chunk(100))
    assert det.last_event is None
    det.process(chunk(3000), listening=False)
    assert det.last_event is None
