"""ElevenLabs language setting, and Yandex STT/TTS wired through the provider registry."""
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from pipecat.services.elevenlabs.dialogue.tts import ElevenLabsDialogueTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.yandex.stt import YandexSTTService
from pipecat.services.yandex.tts import YandexTTSService
from pydantic import TypeAdapter

from api.services.configuration.registry import (
    ElevenlabsTTSConfiguration,
    ServiceProviders,
    STTConfig,
    TTSConfig,
    YandexSTTConfiguration,
    YandexTTSConfiguration,
)
from api.services.pipecat.audio_config import AudioConfig
from api.services.pipecat.service_factory import create_stt_service, create_tts_service

AUDIO = AudioConfig(transport_in_sample_rate=16000, transport_out_sample_rate=16000)


def elevenlabs(model: str, language: str | None = None):
    kwargs = {"api_key": "test-key", "model": model, "voice": "test-voice"}
    if language is not None:
        kwargs["language"] = language
    return create_tts_service(SimpleNamespace(tts=ElevenlabsTTSConfiguration(**kwargs)), AUDIO)


@pytest.mark.parametrize("model", ["eleven_v4_turbo", "eleven_v4"])
def test_v4_with_uzbek_sends_language_code(model):
    service = elevenlabs(model, "uz")
    assert type(service) is ElevenLabsDialogueTTSService
    query = parse_qs(urlparse(service._build_websocket_url()).query)
    assert query["language_code"] == ["uz"]
    assert query["model_id"] == [model]


@pytest.mark.parametrize("language", [None, ""])
def test_v4_without_language_lets_elevenlabs_detect(language):
    service = elevenlabs("eleven_v4_turbo", language)
    assert "language_code" not in parse_qs(urlparse(service._build_websocket_url()).query)


def test_language_field_is_optional_and_documented():
    field = ElevenlabsTTSConfiguration.model_json_schema()["properties"]["language"]
    assert field["default"] == ""
    assert "uz" in field["examples"]
    # configs saved before this field existed still load
    assert ElevenlabsTTSConfiguration(api_key="k").language == ""


def test_older_models_still_use_text_to_speech_and_accept_the_setting():
    service = elevenlabs("eleven_multilingual_v2", "uz")   # takes no language code: ignored, no error
    assert type(service) is ElevenLabsTTSService
    assert "/text-to-speech/" in service._build_websocket_url()


def test_yandex_tts_via_registry_and_factory():
    config = TypeAdapter(TTSConfig).validate_python(
        {"provider": "yandex", "api_key": "k", "folder_id": "folder", "language": "uz-UZ", "voice": ""})
    assert isinstance(config, YandexTTSConfiguration)
    assert config.provider == ServiceProviders.YANDEX
    service = create_tts_service(SimpleNamespace(tts=config), AUDIO)
    assert type(service) is YandexTTSService
    assert service._language == "uz-UZ"
    assert service._folder_id == "folder"


def test_yandex_stt_via_registry_and_factory():
    config = TypeAdapter(STTConfig).validate_python(
        {"provider": "yandex", "api_key": "k", "folder_id": "folder", "language": "uz-UZ"})
    assert isinstance(config, YandexSTTConfiguration)
    service = create_stt_service(SimpleNamespace(stt=config), AUDIO)
    assert type(service) is YandexSTTService


def test_yandex_provider_metadata():
    schema = YandexTTSConfiguration.model_json_schema()
    assert schema["title"] == "Yandex"
    assert "speechkit" in schema.get("provider_docs_url", "")
