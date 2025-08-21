# mcp_ai/mcp_streamlit/config.py
from __future__ import annotations
import os, pathlib
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

try:
    import yaml
except Exception:
    yaml = None  # type: ignore

def _default_output_root() -> str:
    env = os.getenv("OUTPUT_ROOT")
    if env and os.path.isdir(env): return env
    here = pathlib.Path(__file__).resolve().parent
    candidates = [
        here.parent / "output_files",
        here.parent.parent / "mcp_ai" / "output_files",
        here.parent.parent / "mcp_ai-v2" / "mcp_ai" / "output_files",
    ]
    for c in candidates:
        if c.is_dir(): return str(c)
    return "./mcp_ai/output_files"

class Settings(BaseModel):
    OUTPUT_ROOT: str = Field(default_factory=_default_output_root)
    RUN_ID: Optional[str] = Field(default=os.getenv("RUN_ID"))

    OLLAMA_URL: str = Field(default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"))
    OLLAMA_MODEL: str = Field(default=os.getenv("OLLAMA_MODEL", "llama3.1"))
    OLLAMA_EMBED_MODEL: str = Field(default=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

    CLOUD_LLM_ENABLED: bool = False
    CLOUD_LLM_PROVIDER: str = "none"  # "none" | "openai" | "azure-openai"

    AZURE_OPENAI_API_KEY: Optional[str] = Field(default=os.getenv("AZURE_OPENAI_API_KEY"))
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(default=os.getenv("AZURE_OPENAI_ENDPOINT"))
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = Field(default=os.getenv("AZURE_OPENAI_DEPLOYMENT"))
    AZURE_OPENAI_EMBED_DEPLOYMENT: Optional[str] = Field(default=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"))

    OPENAI_API_KEY: Optional[str] = Field(default=os.getenv("OPENAI_API_KEY"))
    OPENAI_MODEL: str = Field(default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    OPENAI_EMBED_MODEL: str = Field(default=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))

    RAG_CHUNK_CHARS: int = 1200
    RAG_OVERLAP: int = 120
    RAG_TOP_K: int = 6

    SCAN_COMMAND: Optional[str] = Field(default=os.getenv("SCAN_COMMAND"))
    CONFIG_FILE: str = Field(default=os.getenv("CONFIG_FILE", str(pathlib.Path(__file__).resolve().parents[1] / "config.yaml")))

    def load_yaml_overrides(self) -> None:
        path = pathlib.Path(self.CONFIG_FILE)
        if yaml is None or not path.exists(): return
        try:
            data: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return

        ai = (data.get("ai") or {})
        local = (ai.get("local") or {})
        cloud = (ai.get("cloud") or {})

        oll = (local.get("ollama") or {})
        self.OLLAMA_URL = str(oll.get("url", self.OLLAMA_URL))
        self.OLLAMA_MODEL = str(oll.get("model", self.OLLAMA_MODEL))
        self.OLLAMA_EMBED_MODEL = str(oll.get("embed_model", self.OLLAMA_EMBED_MODEL))

        if "enabled" in cloud and isinstance(cloud["enabled"], bool):
            self.CLOUD_LLM_ENABLED = cloud["enabled"]
        prov = str(cloud.get("provider", self.CLOUD_LLM_PROVIDER)).lower()
        if prov in ("none","openai","azure-openai"):
            self.CLOUD_LLM_PROVIDER = prov

        az = (cloud.get("azure") or {})
        self.AZURE_OPENAI_API_KEY = az.get("api_key", self.AZURE_OPENAI_API_KEY)
        self.AZURE_OPENAI_ENDPOINT = az.get("endpoint", self.AZURE_OPENAI_ENDPOINT)
        self.AZURE_OPENAI_DEPLOYMENT = az.get("deployment", self.AZURE_OPENAI_DEPLOYMENT)
        self.AZURE_OPENAI_EMBED_DEPLOYMENT = az.get("embed_deployment", self.AZURE_OPENAI_EMBED_DEPLOYMENT)

        oa = (cloud.get("openai") or {})
        self.OPENAI_API_KEY = oa.get("api_key", self.OPENAI_API_KEY)
        self.OPENAI_MODEL = str(oa.get("model", self.OPENAI_MODEL))
        self.OPENAI_EMBED_MODEL = str(oa.get("embed_model", self.OPENAI_EMBED_MODEL))

        scan = (data.get("scan") or {})
        if isinstance(scan.get("command"), str) and scan["command"].strip():
            self.SCAN_COMMAND = scan["command"]

    def cloud_ready(self) -> bool:
        if not self.CLOUD_LLM_ENABLED: return False
        if self.CLOUD_LLM_PROVIDER == "openai":
            return bool(self.OPENAI_API_KEY)
        if self.CLOUD_LLM_PROVIDER == "azure-openai":
            return bool(self.AZURE_OPENAI_API_KEY and self.AZURE_OPENAI_ENDPOINT and self.AZURE_OPENAI_DEPLOYMENT)
        return False

settings = Settings()
settings.load_yaml_overrides()