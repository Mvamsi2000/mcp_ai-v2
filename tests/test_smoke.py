# tests/test_smoke.py
import os

def test_tree_exists():
    assert os.path.exists("mcp_ai/config.yaml")
    assert os.path.exists("mcp_ai/main.py")
    assert os.path.exists("mcp_ai/out")
