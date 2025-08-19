# tests/test_redaction.py
from mcp_ai.redaction import mask

def test_mask():
    txt, counts = mask("email me a@b.com or call 123-456-7890. SSN: 111-22-3333")
    assert counts["email"] >= 1
    assert counts["phone"] >= 1
