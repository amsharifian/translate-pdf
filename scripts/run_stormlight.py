from __future__ import annotations

import subprocess
import sys
import time


def main() -> None:
    python = sys.executable
    worker = subprocess.Popen([python, "scripts/worker.py"])
    try:
        # Give worker a moment to start
        time.sleep(0.5)
        # Run streamlit app (foreground)
        subprocess.run([python, "-m", "streamlit", "run", "app.py"], check=False)
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()


if __name__ == "__main__":
    main()
