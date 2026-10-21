from types import SimpleNamespace
from unittest import mock

import pytest
import requests

from semubot_brain.config import load_config, read_env_file
from semubot_brain.conversation import Conversation
from semubot_brain.llm_backends import BackendError, ClaudeBackend, OpenAIChatBackend


def _response(status=200, payload=None, text=''):
    resp = mock.Mock(status_code=status, text=text)
    if isinstance(payload, Exception):
        resp.json.side_effect = payload
    else:
        resp.json.return_value = payload
    return resp


# --- OpenAI-style backend (llama.cpp / OpenAI) ------------------------------

def test_openai_backend_builds_request_and_parses_reply():
    backend = OpenAIChatBackend('http://127.0.0.1:8082/', 'local',
                                extra_body={'cache_prompt': True})
    ok = _response(payload={'choices': [{'message': {'content': ' Tere! '}}]})
    with mock.patch('requests.post', return_value=ok) as post:
        reply = backend.reply('SYS', [{'role': 'user', 'content': 'Tere'}])
    assert reply == 'Tere!'
    url = post.call_args.args[0]
    body = post.call_args.kwargs['json']
    assert url == 'http://127.0.0.1:8082/v1/chat/completions'
    assert body['messages'][0] == {'role': 'system', 'content': 'SYS'}
    assert body['messages'][1] == {'role': 'user', 'content': 'Tere'}
    assert body['max_tokens'] == 256
    assert body['cache_prompt'] is True
    assert post.call_args.kwargs['headers'] == {}


def test_openai_backend_sends_key_and_token_param():
    backend = OpenAIChatBackend('https://api.openai.com', 'm', api_key='k',
                                max_tokens=1024, token_param='max_completion_tokens')
    ok = _response(payload={'choices': [{'message': {'content': 'x'}}]})
    with mock.patch('requests.post', return_value=ok) as post:
        backend.reply('S', [])
    assert post.call_args.kwargs['headers'] == {'Authorization': 'Bearer k'}
    assert post.call_args.kwargs['json']['max_completion_tokens'] == 1024
    assert 'max_tokens' not in post.call_args.kwargs['json']


@pytest.mark.parametrize('outcome', [
    requests.exceptions.Timeout('slow'),
    requests.exceptions.ConnectionError('down'),
    _response(status=500, text='boom'),
    _response(payload=ValueError('not json')),
    _response(payload={'choices': []}),
    _response(payload={'choices': [{'message': {'content': '  '}}]}),
])
def test_openai_backend_errors_raise_backend_error(outcome):
    backend = OpenAIChatBackend('http://x', 'local')
    kwargs = {'side_effect': outcome} if isinstance(outcome, Exception) else {
        'return_value': outcome}
    with mock.patch('requests.post', **kwargs):
        with pytest.raises(BackendError):
            backend.reply('S', [])


# --- Claude backend ----------------------------------------------------------

def _claude_message(text='Tere!', stop_reason='end_turn'):
    blocks = [SimpleNamespace(type='thinking', thinking=''),
              SimpleNamespace(type='text', text=text)]
    return SimpleNamespace(content=blocks, stop_reason=stop_reason)


def test_claude_backend_uses_low_effort_and_fallbacks():
    client = mock.Mock()
    client.beta.messages.create.return_value = _claude_message(' Tere! ')
    backend = ClaudeBackend('claude-opus-5-5', 'key', client=client)
    msgs = [{'role': 'user', 'content': 'Tere'}]
    assert backend.reply('SYS', msgs) == 'Tere!'
    kwargs = client.beta.messages.create.call_args.kwargs
    assert kwargs['model'] == 'claude-opus-5-5'
    assert kwargs['system'] == 'SYS'
    assert kwargs['messages'] == msgs
    assert kwargs['output_config'] == {'effort': 'low'}
    assert kwargs['fallbacks'] == 'default'
    assert kwargs['betas'] == ['server-side-fallback-2026-07-01']


def test_claude_backend_haiku_uses_plain_messages_api():
    client = mock.Mock()
    client.messages.create.return_value = _claude_message()
    ClaudeBackend('claude-haiku-4-5', 'key', client=client).reply('S', [])
    kwargs = client.messages.create.call_args.kwargs
    assert 'output_config' not in kwargs and 'fallbacks' not in kwargs
    client.beta.messages.create.assert_not_called()


def test_claude_backend_refusal_and_api_error_raise():
    import anthropic
    client = mock.Mock()
    backend = ClaudeBackend('claude-opus-5-5', 'key', client=client)
    client.beta.messages.create.return_value = _claude_message(stop_reason='refusal')
    with pytest.raises(BackendError):
        backend.reply('S', [])
    client.beta.messages.create.side_effect = anthropic.APIConnectionError(
        request=mock.Mock())
    with pytest.raises(BackendError):
        backend.reply('S', [])


# --- Conversation ------------------------------------------------------------

def test_conversation_trims_and_resets():
    backend = mock.Mock()
    backend.reply.side_effect = lambda system, msgs: f'vastus {len(msgs)}'
    conv = Conversation(max_exchanges=2)
    for i in range(3):
        conv.ask(backend, 'S', f'küsimus {i}')
    assert len(conv.messages) == 4
    assert conv.messages[0] == {'role': 'user', 'content': 'küsimus 1'}
    conv.reset()
    assert conv.messages == []


def test_conversation_unchanged_on_backend_error():
    backend = mock.Mock()
    backend.reply.side_effect = BackendError('down')
    conv = Conversation()
    with pytest.raises(BackendError):
        conv.ask(backend, 'S', 'tere')
    assert conv.messages == []


# --- Config ------------------------------------------------------------------

def test_config_defaults_and_missing_claude_key():
    config = load_config({})
    assert config.whisper_url == 'http://127.0.0.1:8081/inference'
    assert config.tts_url == 'http://127.0.0.1:8001/v2'
    assert config.tts_speaker == 'mari'
    assert config.local_backend.url == 'http://127.0.0.1:8082/v1/chat/completions'
    assert config.cloud_backend is None
    assert 'ANTHROPIC_API_KEY' in config.cloud_error


def test_config_claude_default_model():
    config = load_config({'ANTHROPIC_API_KEY': 'k'})
    assert isinstance(config.cloud_backend, ClaudeBackend)
    assert config.cloud_backend.model == 'claude-opus-5-5'


def test_config_openai_needs_model():
    assert load_config({'CLOUD_PROVIDER': 'openai', 'OPENAI_API_KEY': 'k'}).cloud_backend is None
    config = load_config({'CLOUD_PROVIDER': 'openai', 'OPENAI_API_KEY': 'k',
                          'OPENAI_MODEL': 'some-model'})
    assert config.cloud_backend.url == 'https://api.openai.com/v1/chat/completions'
    assert config.cloud_backend.model == 'some-model'


def test_config_unknown_provider():
    config = load_config({'CLOUD_PROVIDER': 'gemini'})
    assert config.cloud_backend is None
    assert 'gemini' in config.cloud_error


def test_read_env_file(tmp_path):
    env = tmp_path / 'brain.env'
    env.write_text('# comment\n\nCLOUD_PROVIDER=claude   # claude | openai\n'
                   'ANTHROPIC_API_KEY="sk-x"\nOPENAI_MODEL=\nBROKEN LINE\n')
    assert read_env_file(env) == {'CLOUD_PROVIDER': 'claude', 'ANTHROPIC_API_KEY': 'sk-x',
                                  'OPENAI_MODEL': ''}
    assert read_env_file(tmp_path / 'missing.env') == {}
