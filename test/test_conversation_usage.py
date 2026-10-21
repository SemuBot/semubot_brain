import datetime
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from semubot_brain.conversation import Conversation, ConversationLog
from semubot_brain.llm_backends import BackendError, ClaudeBackend, OpenAIChatBackend
from semubot_brain.usage import UsageTracker, cost, price_for


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_history_cleared_after_five_minutes_of_silence():
    clock = Clock()
    conv = Conversation(idle_reset=300, clock=clock)
    backend = mock.Mock()
    backend.reply.return_value = 'vastus'
    conv.ask(backend, 'S', 'tere')
    clock.now += 299
    assert not conv.expire_if_idle() and len(conv.messages) == 2
    conv.ask(backend, 'S', 'veel')  # activity restarts the 5 minutes
    clock.now += 299
    assert not conv.expire_if_idle()
    clock.now += 1
    assert conv.expire_if_idle() and conv.messages == []
    assert not conv.expire_if_idle()  # nothing left to clear


def test_ask_after_long_silence_starts_fresh():
    clock = Clock()
    conv = Conversation(idle_reset=300, clock=clock)
    backend = mock.Mock()
    backend.reply.return_value = 'vastus'
    conv.ask(backend, 'S', 'esimene')
    clock.now += 600
    conv.ask(backend, 'S', 'teine')
    assert backend.reply.call_args.args[1] == [{'role': 'user', 'content': 'teine'}]


def test_log_keeps_last_entries_and_rejects_unknown_kinds():
    log = ConversationLog(size=3, clock=lambda: 12.34)
    for i in range(5):
        log.add('heard', f'lause {i}')
    assert [e['text'] for e in log.as_list()] == ['lause 2', 'lause 3', 'lause 4']
    assert log.as_list()[0] == {'time': 12.3, 'kind': 'heard', 'text': 'lause 2'}
    with pytest.raises(ValueError):
        log.add('secret', 'x')
    log.clear()
    assert log.as_list() == []


def test_cost_and_prices():
    assert price_for('claude-opus-5-5', {}) == (4.0, 20.0)
    assert price_for('gpt-x', {}) is None
    assert price_for('gpt-x', {'CLOUD_PRICE_INPUT': '1.5', 'CLOUD_PRICE_OUTPUT': '6'}) == (1.5, 6.0)
    usage = {'input_tokens': 1_000_000, 'output_tokens': 100_000,
             'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 1_000_000}
    assert cost(usage, (4.0, 20.0)) == pytest.approx(4.0 + 2.0 + 0.4)
    assert cost(usage, None) is None


def test_usage_tracker_counts_per_day_and_month(tmp_path):
    day = [datetime.date(2026, 10, 5)]
    path = str(tmp_path / 'cloud_usage.json')
    tracker = UsageTracker(path, today=lambda: day[0])
    tracker.record({'input_tokens': 900, 'output_tokens': 50}, (4.0, 20.0))
    tracker.record({'input_tokens': 100, 'output_tokens': 50, 'cache_read_input_tokens': 10},
                   (4.0, 20.0))
    tracker.record({'input_tokens': 10, 'output_tokens': 10}, None)
    today = tracker.summary()['today']
    assert today['requests'] == 3 and today['input_tokens'] == 1020
    assert today['output_tokens'] == 110 and today['unpriced'] == 1
    assert today['cost'] == pytest.approx((1000 * 4 + 10 * 0.4 + 100 * 20) / 1e6)
    day[0] = datetime.date(2026, 10, 6)
    tracker.record({'input_tokens': 1, 'output_tokens': 1}, (4.0, 20.0))
    assert tracker.summary()['today']['requests'] == 1
    assert tracker.summary()['month']['requests'] == 4
    reloaded = UsageTracker(path, today=lambda: day[0])
    assert reloaded.summary() == tracker.summary()
    assert set(json.load(open(path))) == {'2026-10-05', '2026-10-06'}


def test_usage_tracker_keeps_about_two_months(tmp_path):
    day = [datetime.date(2026, 1, 1)]
    tracker = UsageTracker(str(tmp_path / 'u.json'), today=lambda: day[0])
    for offset in range(70):
        day[0] = datetime.date(2026, 1, 1) + datetime.timedelta(days=offset)
        tracker.record({'input_tokens': 1, 'output_tokens': 1}, None)
    assert len(tracker.days) == 62 and min(tracker.days) == '2026-01-09'


def test_backends_report_token_usage():
    backend = OpenAIChatBackend('http://x', 'm')
    ok = mock.Mock(status_code=200)
    ok.json.return_value = {'choices': [{'message': {'content': 'hei'}}],
                            'usage': {'prompt_tokens': 120, 'completion_tokens': 8}}
    with mock.patch('requests.post', return_value=ok):
        backend.reply('S', [])
    assert backend.last_usage == {'input_tokens': 120, 'output_tokens': 8}

    client = mock.Mock()
    usage = SimpleNamespace(input_tokens=300, output_tokens=40, cache_creation_input_tokens=None,
                            cache_read_input_tokens=0)
    client.beta.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type='text', text='Tere')], stop_reason='refusal', usage=usage)
    claude = ClaudeBackend('claude-opus-5-5', 'k', client=client)
    with pytest.raises(BackendError):
        claude.reply('S', [])
    assert claude.last_usage == {'input_tokens': 300, 'output_tokens': 40,
                                 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0}


def test_log_entries_can_carry_step_timings():
    log = ConversationLog(clock=lambda: 5.0)
    log.add('said', 'Tere!', 'Kohalik AI 4.2 s · kõnesüntees 0.9 s')
    log.add('heard', 'tere', None)
    assert log.as_list() == [
        {'time': 5.0, 'kind': 'said', 'text': 'Tere!',
         'details': 'Kohalik AI 4.2 s · kõnesüntees 0.9 s'},
        {'time': 5.0, 'kind': 'heard', 'text': 'tere'},
    ]
