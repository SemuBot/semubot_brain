"""LLM backends used by the brain node. No ROS dependency, so they are unit-testable."""
import requests


class BackendError(Exception):
    """Raised when a backend cannot produce a reply."""


class OpenAIChatBackend:
    """OpenAI-style /v1/chat/completions client (llama.cpp server or OpenAI)."""

    def __init__(self, base_url, model, api_key=None, max_tokens=256,
                 token_param='max_tokens', extra_body=None, timeout=60):
        self.url = base_url.rstrip('/') + '/v1/chat/completions'
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.token_param = token_param
        self.extra_body = extra_body or {}
        self.timeout = timeout
        self.last_usage = None  # Token counts of the last successful request

    def reply(self, system_prompt, messages):
        body = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': system_prompt}] + list(messages),
            self.token_param: self.max_tokens,
            **self.extra_body,
        }
        headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
        self.last_usage = None
        try:
            response = requests.post(self.url, json=body, headers=headers, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            raise BackendError(f'{self.url}: {e}') from e
        if response.status_code != 200:
            raise BackendError(f'{self.url}: HTTP {response.status_code}: {response.text[:300]}')
        try:
            text = response.json()['choices'][0]['message']['content']
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise BackendError(f'{self.url}: unexpected response: {response.text[:300]}') from e
        if not isinstance(text, str) or not text.strip():
            raise BackendError(f'{self.url}: empty reply')
        usage = response.json().get('usage') or {}
        self.last_usage = {'input_tokens': usage.get('prompt_tokens', 0) or 0,
                           'output_tokens': usage.get('completion_tokens', 0) or 0}
        return text.strip()


class ClaudeBackend:
    """Claude via the official Anthropic SDK."""

    def __init__(self, model, api_key, timeout=60, client=None):
        import anthropic
        self._anthropic = anthropic
        self.model = model
        self.client = client or anthropic.Anthropic(
            api_key=api_key, timeout=timeout, max_retries=1)
        self.last_usage = None  # Token counts of the last successful request

    def reply(self, system_prompt, messages):
        kwargs = {
            'model': self.model,
            'system': system_prompt,
            'messages': list(messages),
        }
        if self.model.startswith('claude-haiku'):
            # Haiku 4.5: no effort parameter, no server-side fallbacks.
            create = self.client.messages.create
            kwargs['max_tokens'] = 512
        else:
            # Current models think by default; low effort keeps voice replies fast.
            create = self.client.beta.messages.create
            kwargs.update(
                max_tokens=4096,
                output_config={'effort': 'low'},
                betas=['server-side-fallback-2026-07-01'],
                fallbacks='default',
            )
        self.last_usage = None
        try:
            response = create(**kwargs)
        except self._anthropic.APIError as e:
            raise BackendError(f'Claude API: {e}') from e
        # Billed even when the reply is refused or empty, so count it first.
        usage = getattr(response, 'usage', None)
        self.last_usage = {key: getattr(usage, key, 0) or 0 for key in (
            'input_tokens', 'output_tokens',
            'cache_creation_input_tokens', 'cache_read_input_tokens')}
        if response.stop_reason == 'refusal':
            raise BackendError('Claude API: request refused')
        text = ''.join(b.text for b in response.content if b.type == 'text').strip()
        if not text:
            raise BackendError(f'Claude API: empty reply (stop_reason={response.stop_reason})')
        return text
