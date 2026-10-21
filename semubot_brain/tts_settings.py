"""Robot voice settings (TartuNLP speaker and speed), shared through ~/.config/semubot/tts.json."""
import json
import os

SETTINGS_FILE = os.path.expanduser('~/.config/semubot/tts.json')
MIN_SPEED = 0.5  # Limits of the TartuNLP TTS API
MAX_SPEED = 2.0


def validate(settings, speakers, current):
    """Return a clean {'speaker', 'speed'} dict from a partial update, or raise ValueError."""
    if not isinstance(settings, dict):
        raise ValueError('settings must be a JSON object')
    speaker = settings.get('speaker', current['speaker'])
    if not isinstance(speaker, str) or not speaker:
        raise ValueError('speaker must be a name')
    if speakers and speaker not in speakers:
        raise ValueError(f'unknown speaker {speaker!r}')
    speed = settings.get('speed', current['speed'])
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise ValueError('speed must be a number')
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(f'speed must be between {MIN_SPEED} and {MAX_SPEED}')
    return {'speaker': speaker, 'speed': round(float(speed), 2)}


def load(path, default_speaker):
    defaults = {'speaker': default_speaker, 'speed': 1.0}
    try:
        with open(path, encoding='utf-8') as file:
            return validate(json.load(file), [], defaults)
    except (OSError, ValueError):
        return defaults


def save(path, settings):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as file:
        json.dump(settings, file)
    os.replace(temporary, path)
