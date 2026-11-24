import subprocess
import sys
import time
from pathlib import Path

Path("RESULTS").mkdir(exist_ok=True)

for mode in ["baseline", "optimized"]:
    print(f"\nMonitoring {mode.upper()} GPU workload...")

    filename = Path("RESULTS") / f"smi_{mode}.csv"

    with open(filename, "w", encoding="utf-8") as log:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used",
                "--format=csv",
                "-lms", "100"
            ],
            stdout=log
        )

        try:
            time.sleep(0.5)

            subprocess.run(
                [
                    sys.executable,
                    "gpu_workload.py",
                    "--mode", mode
                ],
                check=True
            )

        finally:
            monitor.terminate()
            monitor.wait()

    print(f"Saved: {filename}")

print("\nGPU monitoring completed!")
