# tests/test_router.py
from mcp_ai.llm_agent import fast_pass, deep_pass_local

def test_fast_pass():
    out = fast_pass("This is an invoice total $100")
    assert "category" in out
    assert "confidence" in out

def test_deep_pass():
    out = deep_pass_local("Please contact me at 123-456-7890 or email x@y.com.")
    assert "pii" in out
