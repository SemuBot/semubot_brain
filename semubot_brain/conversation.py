"""Chat history kept between turns, plus a short log of the conversation for the chest display."""
import collections
import time


class Conversation:

    def __init__(self, max_exchanges=6, idle_reset=300.0, clock=time.monotonic):
        self.max_exchanges = max_exchanges
        self.idle_reset = idle_reset  # Seconds without an exchange before a new conversation
        self.clock = clock
        self.messages = []
        self.last_activity = clock()

    def reset(self):
        self.messages = []

    def expire_if_idle(self):
        """Forget the history after idle_reset seconds of silence; True if it was forgotten."""
        if self.messages and self.clock() - self.last_activity >= self.idle_reset:
            self.reset()
            return True
        return False

    def ask(self, backend, system_prompt, user_text):
        """Get a reply for user_text; history only changes if the backend succeeds."""
        self.expire_if_idle()
        user_message = {'role': 'user', 'content': user_text}
        reply = backend.reply(system_prompt, self.messages + [user_message])
        self.messages += [user_message, {'role': 'assistant', 'content': reply}]
        self.messages = self.messages[-2 * self.max_exchanges:]
        self.last_activity = self.clock()
        return reply


class ConversationLog:
    """Last few things heard and said. Kept in memory only, never written to disk."""

    KINDS = ('heard', 'said', 'typed', 'info', 'error', 'audio')  # audio: recording events

    def __init__(self, size=30, clock=time.time):
        self.entries = collections.deque(maxlen=size)
        self.clock = clock

    def add(self, kind, text, details=None):
        """details: optional short note, e.g. how long each step took."""
        if kind not in self.KINDS:
            raise ValueError(f'unknown log entry kind {kind!r}')
        entry = {'time': round(self.clock(), 1), 'kind': kind, 'text': text}
        if details:
            entry['details'] = details
        self.entries.append(entry)

    def clear(self):
        self.entries.clear()

    def as_list(self):
        return list(self.entries)
