from src.production.serving import ForecastRequest


def test_api_request_requires_eight_observations():
    try:
        ForecastRequest(observations=[])
    except Exception as exc:
        assert "observations" in str(exc)
    else:
        raise AssertionError("request without observations must be rejected")
