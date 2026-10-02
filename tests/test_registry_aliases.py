from src.production.registry import _promote_model_alias


class FakeVersion:
    def __init__(self, version):
        self.version = str(version)


class FakeClient:
    def __init__(self, aliases):
        self.aliases = {name: {alias: FakeVersion(version) for alias, version in values.items()} for name, values in aliases.items()}
        self.calls = []

    def get_model_version_by_alias(self, name, alias):
        if alias not in self.aliases.get(name, {}):
            raise RuntimeError("alias not found")
        return self.aliases[name][alias]

    def set_registered_model_alias(self, name, alias, version):
        self.aliases.setdefault(name, {})[alias] = FakeVersion(version)
        self.calls.append(("set", name, alias, str(version)))

    def delete_registered_model_alias(self, name, alias):
        self.aliases.setdefault(name, {}).pop(alias, None)
        self.calls.append(("delete", name, alias))


def test_new_version_moves_old_champion_to_previous():
    client = FakeClient({"weather-h1": {"champion": "2", "candidate": "3"}})

    _promote_model_alias(client, "weather-h1", "3")

    assert client.aliases["weather-h1"]["champion"].version == "3"
    assert client.aliases["weather-h1"]["previous"].version == "2"
    assert "candidate" not in client.aliases["weather-h1"]


def test_repromoting_champion_does_not_assign_previous_to_same_version():
    client = FakeClient({"weather-h1": {"champion": "3", "previous": "3", "candidate": "3"}})

    _promote_model_alias(client, "weather-h1", "3")

    assert client.aliases["weather-h1"]["champion"].version == "3"
    assert "previous" not in client.aliases["weather-h1"]
    assert "candidate" not in client.aliases["weather-h1"]
