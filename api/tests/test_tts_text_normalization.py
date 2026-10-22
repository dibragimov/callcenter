"""The 'text_normalization' TTS setting adds the Uzbek normalizer filter for every provider."""
from types import SimpleNamespace

import pytest
from pipecat.utils.text.uzbek_text_filter import UzbekTextNormalizerFilter
from pipecat.utils.text.xml_function_tag_filter import XMLFunctionTagFilter
from pydantic import TypeAdapter

from api.services.configuration.model_connections import model_connection_catalog
from api.services.configuration.registry import TTSConfig
from api.services.pipecat.audio_config import AudioConfig
from api.services.pipecat.service_factory import create_tts_service

AUDIO = AudioConfig(transport_in_sample_rate=16000, transport_out_sample_rate=16000)
PROVIDERS = {
    "yandex": {"provider": "yandex", "api_key": "k", "folder_id": "f", "language": "uz-UZ"},
    "elevenlabs-v4": {"provider": "elevenlabs", "api_key": "k", "model": "eleven_v4_turbo", "voice": "v"},
    "elevenlabs-flash": {"provider": "elevenlabs", "api_key": "k", "model": "eleven_flash_v2_5", "voice": "v"},
    "speaches": {"provider": "speaches", "base_url": "https://tts.example.com/v1", "model": "navoiy-tts-v1",
                 "voice": "calm"},
    "deepgram": {"provider": "deepgram", "api_key": "k"},
    "xai": {"provider": "xai", "api_key": "k"},
}


def filters(raw: dict):
    config = TypeAdapter(TTSConfig).validate_python(raw)
    return [type(f) for f in create_tts_service(SimpleNamespace(tts=config), AUDIO)._text_filters]


@pytest.mark.parametrize("name", PROVIDERS)
def test_uzbek_adds_the_filter(name):
    got = filters({**PROVIDERS[name], "text_normalization": "uzbek"})
    assert got == [XMLFunctionTagFilter, UzbekTextNormalizerFilter]


@pytest.mark.parametrize("name", PROVIDERS)
def test_off_and_saved_configs_without_the_field_are_unchanged(name):
    assert filters({**PROVIDERS[name], "text_normalization": "off"}) == [XMLFunctionTagFilter]
    assert filters(PROVIDERS[name]) == [XMLFunctionTagFilter]


def test_invalid_value_is_rejected():
    with pytest.raises(Exception):
        TypeAdapter(TTSConfig).validate_python({**PROVIDERS["yandex"], "text_normalization": "kazakh"})


def test_every_tts_provider_shows_the_setting_in_the_model_ui():
    tts = model_connection_catalog()["services"]["tts"]
    assert len(tts) > 5
    for provider, entry in tts.items():
        field = entry["settings_schema"]["properties"].get("text_normalization")
        assert field is not None, provider
        assert field.get("enum") == ["off", "uzbek"] and field.get("default") == "off", provider


def test_setting_is_last_so_it_never_takes_the_primary_slot():
    tts = model_connection_catalog()["services"]["tts"]
    for provider, entry in tts.items():
        assert list(entry["settings_schema"]["properties"])[-1] == "text_normalization", provider
