from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_server_import_without_langserve_in_dev_mode():
    backend_root = Path(__file__).resolve().parents[1]
    script = """
import builtins
import os
import sys
from pathlib import Path

repo_backend = Path(r'__BACKEND_ROOT__')
sys.path.insert(0, str(repo_backend))
os.environ['APP_ENV'] = 'development'

_orig_import = builtins.__import__
def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'langserve' or name.startswith('langserve.'):
        raise ImportError('langserve intentionally missing for test')
    return _orig_import(name, globals, locals, fromlist, level)

builtins.__import__ = _blocked_import
import src.server as server
assert server.app is not None
print('ok')
"""
    script = script.replace("__BACKEND_ROOT__", str(backend_root).replace("\\", "\\\\"))
    env = dict(os.environ)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(backend_root),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
