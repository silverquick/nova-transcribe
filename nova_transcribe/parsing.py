import json
import re
from typing import Optional


def _extract_generation_stage(content_start_event: dict) -> Optional[str]:
    amf = content_start_event.get("additionalModelFields")
    if not amf:
        return None
    try:
        return json.loads(amf).get("generationStage")
    except Exception:
        return None


_INTERRUPTED_TAG_RE = re.compile(r"^\s*\{\s*[\"']interrupted[\"']\s*:\s*true\s*\}\s*$", re.I)


def _is_interrupted_tag(text: str) -> bool:
    if not text:
        return False
    if _INTERRUPTED_TAG_RE.match(text):
        return True
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False
    try:
        obj = json.loads(stripped)
    except Exception:
        return False
    return isinstance(obj, dict) and obj.get("interrupted") is True and len(obj) == 1


_INDEXED_LINE_RE = re.compile(r"^\s*(\d{1,6})\s*[\t:]\s*(.*?)\s*$")


def _parse_indexed_output_line(line: str) -> Optional[tuple[int, str]]:
    """
    例: "1\tこんにちは" / "1: こんにちは"
    """
    if not line:
        return None
    line = line.strip("\r")
    if not line.strip():
        return None
    m = _INDEXED_LINE_RE.match(line)
    if not m:
        return None
    idx = int(m.group(1))
    ja = m.group(2).strip()
    if not ja:
        return None
    return idx, ja


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    LLM の出力から最初の JSON オブジェクト部分を抜き出す（前後に余計な文字が付くケース対策）。
    """
    if not text:
        return None
    decoder = json.JSONDecoder()
    in_string = False
    escape = False
    depth = 0
    start = None

    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = idx
                try:
                    obj, end = decoder.raw_decode(text[start:])
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(obj, dict):
                        return text[start : start + end]
            depth += 1
            continue

        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : idx + 1]

    return None
