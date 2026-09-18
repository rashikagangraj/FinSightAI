from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.core.config import Settings
from src.llm.gemini_client import GeminiClient


def test_gemini_client_requires_key():
    with patch("src.llm.gemini_client.get_settings") as mock_settings:
        mock_settings.return_value = Settings(gemini_api_key="")
        with pytest.raises(ValueError, match="Gemini API key is missing"):
            GeminiClient()


def test_gemini_client_complete():
    with patch("src.llm.gemini_client.get_settings") as mock_settings, \
         patch("src.llm.gemini_client.OpenAI") as mock_openai:
        mock_settings.return_value = Settings(
            gemini_api_key="test-gemini-key",
            gemini_model="gemini-1.5-flash",
        )
        mock_instance = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Gemini financial answer"
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice],
            usage=MagicMock(total_tokens=15),
        )
        mock_openai.return_value = mock_instance

        client = GeminiClient()
        ans = client.complete("What is the ROI?")
        assert ans == "Gemini financial answer"
        mock_instance.chat.completions.create.assert_called_once()
