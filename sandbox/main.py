import os
import sys
import subprocess
import tempfile

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

_SAFE_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/tmp",
    "PYTHONPATH": "",
}


class RunRequest(BaseModel):
    code: str
    timeout: int = 30


@app.post("/run")
def run_code(body: RunRequest):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(body.code)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=body.timeout,
            cwd="/tmp",
            env=_SAFE_ENV,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": (
                None
                if result.returncode == 0
                else f"exited with code {result.returncode}"
            ),
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "",
            "error": f"timed out after {body.timeout}s",
        }
    finally:
        os.unlink(script_path)
