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
        [sys.executable, "-m", "pytest", "tests"],
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

    ok = True
    for name, code, output in results:
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        print(output)
        ok &= code == 0

    print(f"\n{'PASSED' if ok else 'FAILED'}: {', '.join(n for n, c, _ in results if c != 0) or 'all projects'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
