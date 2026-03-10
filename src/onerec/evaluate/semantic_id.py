from __future__ import annotations

import re


SEMANTIC_ID_PATTERN = re.compile(r"<a_\d+><b_\d+><c_\d+>")


def canonicalize_semantic_id(text: object) -> str:
    if text is None:
        return ""
    value = str(text).strip(" \n\r\t\"")
    match = SEMANTIC_ID_PATTERN.search(value)
    if match:
        return match.group(0)
    return value
