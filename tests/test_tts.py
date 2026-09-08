from app.lilly import BadInput
from app import tts

import pytest


def test_blank_text_is_the_callers_problem(tmp_path):
    with pytest.raises(BadInput):
        tts.speak_to_file("   ", str(tmp_path / "out.wav"))
