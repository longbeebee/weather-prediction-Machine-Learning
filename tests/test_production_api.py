from src.production.serving import ForecastRequest, ModelService


def test_api_request_requires_eight_observations():
    try:
        ForecastRequest(observations=[])
    except Exception as exc:
        assert "observations" in str(exc)
    else:
        raise AssertionError("request without observations must be rejected")


def test_registry_uri_pins_registered_version_for_canary():
    service = ModelService.__new__(ModelService)
    service.model_alias = "candidate"
    assert service._registry_uri({"name": "weather-7d-temperature-h1", "version": "7"}) == "models:/weather-7d-temperature-h1/7"
    assert service._registry_uri({"name": "weather-7d-temperature-h1"}) == "models:/weather-7d-temperature-h1@candidate"
