from types import SimpleNamespace


class _FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=item))])


class FakeGroq:
    def __init__(self, replies):
        self.chat = SimpleNamespace(completions=_FakeCompletions(replies))

    @property
    def calls(self):
        return self.chat.completions.calls
