import pytest

from semubot_brain import tts_settings

CURRENT = {'speaker': 'mari', 'speed': 1.0}
SPEAKERS = ['albert', 'mari', 'vesta']


def test_partial_updates_keep_the_other_value():
    assert tts_settings.validate({'speaker': 'albert'}, SPEAKERS, CURRENT) == {
        'speaker': 'albert', 'speed': 1.0}
    assert tts_settings.validate({'speed': 1.25}, SPEAKERS, CURRENT) == {
        'speaker': 'mari', 'speed': 1.25}


@pytest.mark.parametrize('settings', [
    {'speaker': 'nobody'}, {'speaker': ''}, {'speaker': 5},
    {'speed': 0.4}, {'speed': 2.1}, {'speed': 'fast'}, {'speed': True}, ['mari'],
])
def test_invalid_settings_raise(settings):
    with pytest.raises(ValueError):
        tts_settings.validate(settings, SPEAKERS, CURRENT)


def test_any_speaker_allowed_when_list_unknown():
    assert tts_settings.validate({'speaker': 'x'}, [], CURRENT)['speaker'] == 'x'


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / 'sub' / 'tts.json')
    assert tts_settings.load(path, 'mari') == {'speaker': 'mari', 'speed': 1.0}
    tts_settings.save(path, {'speaker': 'vesta', 'speed': 0.9})
    assert tts_settings.load(path, 'mari') == {'speaker': 'vesta', 'speed': 0.9}


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / 'tts.json'
    path.write_text('{"speaker": "mari", "speed": 9}')
    assert tts_settings.load(str(path), 'albert') == {'speaker': 'albert', 'speed': 1.0}
