import xml.etree.ElementTree as ET

from signspeak.tts import build_ssml


def test_ssml_escapes_user_text():
    ssml = build_ssml("a < b & 'c' </voice>", "en-IN-NeerjaNeural")
    root = ET.fromstring(ssml)  # must still be valid XML
    voice = root.find("voice")
    assert voice.get("name") == "en-IN-NeerjaNeural"
    assert voice.text == "a < b & 'c' </voice>"
    assert root.get("{http://www.w3.org/XML/1998/namespace}lang") == "en-IN"
