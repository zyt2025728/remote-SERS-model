"""Convert the two tracked Jupytext percent scripts into local notebooks."""
from pathlib import Path

import jupytext


ROOT = Path(__file__).resolve().parent
SOURCES = (ROOT / "jupyter" / "Level2_Jupyter.py", ROOT / "jupyter" / "Level3_Jupyter.py")


def main() -> None:
    for source in SOURCES:
        notebook = jupytext.read(source, fmt="py:percent")
        destination = source.with_suffix(".ipynb")
        jupytext.write(notebook, destination, fmt="ipynb")
        print(f"Created {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
