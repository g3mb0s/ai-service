#!/usr/bin/env python3
"""Запуск AI service через Uvicorn."""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

os.chdir(project_root)

if __name__ == "__main__":
    import uvicorn
    from src.main import app
    
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info",
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
