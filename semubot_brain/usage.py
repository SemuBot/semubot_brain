"""Daily count of cloud LLM requests, tokens and estimated cost (~/.config/semubot/cloud_usage.json)."""
import datetime
import json
import os

USAGE_FILE = os.path.expanduser('~/.config/semubot/cloud_usage.json')
KEEP_DAYS = 62

# USD per million input / output tokens (Anthropic list prices). Other models: set
# CLOUD_PRICE_INPUT / CLOUD_PRICE_OUTPUT in brain.env, otherwise only tokens are counted.
PRICES = {
    'claude-opus-5-5': (4.0, 20.0),
    'claude-sonnet-5-5': (2.0, 10.0),
    'claude-haiku-4-5': (1.0, 5.0),
}
CACHE_WRITE_FACTOR = 1.25
CACHE_READ_FACTOR = 0.1


def price_for(model, env):
    try:
        return float(env['CLOUD_PRICE_INPUT']), float(env['CLOUD_PRICE_OUTPUT'])
    except (KeyError, ValueError):
        return PRICES.get(model)


def cost(usage, prices):
    if prices is None:
        return None
    input_price, output_price = prices
    return (usage.get('input_tokens', 0) * input_price
            + usage.get('cache_creation_input_tokens', 0) * input_price * CACHE_WRITE_FACTOR
            + usage.get('cache_read_input_tokens', 0) * input_price * CACHE_READ_FACTOR
            + usage.get('output_tokens', 0) * output_price) / 1_000_000


class UsageTracker:

    def __init__(self, path=USAGE_FILE, today=datetime.date.today):
        self.path = path
        self.today = today
        try:
            with open(path, encoding='utf-8') as file:
                self.days = json.load(file)
            if not isinstance(self.days, dict):
                self.days = {}
        except (OSError, ValueError):
            self.days = {}

    def record(self, usage, prices):
        """Add one request. usage: token counts as reported by the API; prices: (in, out) or None."""
        day = self.days.setdefault(self.today().isoformat(), {
            'requests': 0, 'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0, 'unpriced': 0})
        day['requests'] += 1
        day['input_tokens'] += (usage.get('input_tokens', 0)
                                + usage.get('cache_creation_input_tokens', 0)
                                + usage.get('cache_read_input_tokens', 0))
        day['output_tokens'] += usage.get('output_tokens', 0)
        request_cost = cost(usage, prices)
        if request_cost is None:
            day['unpriced'] += 1
        else:
            day['cost'] = round(day['cost'] + request_cost, 6)
        for old in sorted(self.days)[:-KEEP_DAYS]:
            del self.days[old]
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        temporary = self.path + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as file:
            json.dump(self.days, file)
        os.replace(temporary, self.path)

    def summary(self):
        today = self.today()
        empty = {'requests': 0, 'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0, 'unpriced': 0}
        month = dict(empty)
        for date, day in self.days.items():
            if date.startswith(today.strftime('%Y-%m')):
                for key in month:
                    month[key] += day.get(key, 0)
        month['cost'] = round(month['cost'], 6)
        return {'today': self.days.get(today.isoformat(), empty), 'month': month}
