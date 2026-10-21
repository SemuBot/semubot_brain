"""Brain configuration from environment variables (see brain.env)."""
import os
from dataclasses import dataclass

from semubot_brain.llm_backends import ClaudeBackend, OpenAIChatBackend
from semubot_brain.usage import price_for

MODES = ('manual', 'local', 'cloud')
ENV_FILE = os.path.expanduser('~/.config/semubot/brain.env')


def read_env_file(path):
    """Parse KEY=VALUE lines; blank lines and # comments are ignored."""
    values = {}
    try:
        with open(path, encoding='utf-8') as file:
            lines = file.read().splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        line = line.split(' #', 1)[0].strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        values[key.strip()] = value.strip().strip('"\'')
    return values


@dataclass
class Config:
    whisper_url: str
    tts_url: str
    tts_speaker: str
    local_backend: object
    cloud_backend: object  # None when misconfigured
    cloud_error: str  # why cloud_backend is None
    vad_model: str
    cloud_model: str = ''
    cloud_prices: tuple = None  # USD per million (input, output) tokens, None if unknown


def load_config(env):
    local = OpenAIChatBackend(
        env.get('LOCAL_LLM_URL', 'http://127.0.0.1:8082'),
        model='local',
        extra_body={'cache_prompt': True},
        timeout=180,  # an uncached system prompt takes ~50 s on the NUC CPU
    )
    cloud, error = _cloud_backend(env)
    cloud_model = getattr(cloud, 'model', '')
    return Config(
        cloud_model=cloud_model,
        cloud_prices=price_for(cloud_model, env) if cloud else None,
        whisper_url=env.get('WHISPER_URL', 'http://127.0.0.1:8081/inference'),
        tts_url=env.get('TTS_URL', 'http://127.0.0.1:8001/v2'),
        tts_speaker=env.get('TTS_SPEAKER', 'mari'),
        local_backend=local,
        cloud_backend=cloud,
        cloud_error=error,
        vad_model=os.path.expanduser(
            env.get('VAD_MODEL', '~/.local/share/semubot/silero_vad.onnx')),
    )


def _cloud_backend(env):
    provider = env.get('CLOUD_PROVIDER', 'claude').strip().lower()
    if provider == 'claude':
        key = env.get('ANTHROPIC_API_KEY', '').strip()
        if not key:
            return None, 'CLOUD_PROVIDER=claude but ANTHROPIC_API_KEY is not set'
        model = env.get('CLAUDE_MODEL', '').strip() or 'claude-opus-5-5'
        return ClaudeBackend(model, key), ''
    if provider == 'openai':
        key = env.get('OPENAI_API_KEY', '').strip()
        model = env.get('OPENAI_MODEL', '').strip()
        if not key or not model:
            return None, 'CLOUD_PROVIDER=openai needs OPENAI_API_KEY and OPENAI_MODEL'
        return OpenAIChatBackend(
            'https://api.openai.com', model, api_key=key,
            max_tokens=1024, token_param='max_completion_tokens'), ''
    return None, f'Unknown CLOUD_PROVIDER {provider!r} (use claude or openai)'
