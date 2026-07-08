"""Run each project's pytest suite in its own process, in parallel.

Separate processes (not one pytest run over both dirs) because both
projects define same-named modules (config.py, model.py, ...) that
would collide in a shared sys.modules cache.
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
PROJECTS = [p.parent for p in ROOT.glob("*/tests") if p.is_dir()]


def run(project: Path) -> tuple[str, int, str]:
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests",
            "-v",
            "--cov=.", "--cov-report=term-missing",
        ],
        cwd=project,
        capture_output=True,
        text=True,
    )
    return project.name, result.returncode, result.stdout + result.stderr


def main() -> int:
    if not PROJECTS:
        print("No projects with a tests/ dir found.")
        return 1

    with ThreadPoolExecutor(max_workers=len(PROJECTS)) as pool:
        results = list(pool.map(run, PROJECTS))

    for name, _, output in results:
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        print(output)

    ok = all(code == 0 for _, code, _ in results)
    print(f"\n{'=' * 20} SUMMARY {'=' * 20}")
    for name, code, _ in results:
        print(f"  {name:<20} {'PASSED' if code == 0 else 'FAILED'}")
    print(f"\n{'ALL PROJECTS PASSED' if ok else 'SOME PROJECTS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
