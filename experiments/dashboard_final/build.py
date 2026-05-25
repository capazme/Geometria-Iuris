"""Entry point — regenerate the six dashboard HTML files in `output/`.

Run from anywhere:

    python3 experiments/dashboard_final/build.py

This:
  1. Imports the six `pages/*.py` modules and calls `build()` on each
  2. Writes the resulting HTML to `output/{home, methodology, how_it_works,
     experiment_31, experiment_32, robustness_caveats}.html`
  3. Copies `assets/plotly.min.js` into `output/assets/plotly.min.js`
  4. Materialises `shared_ui.CSS_MAIN` into `output/assets/style.css`
  5. Reports the size of every artefact

The resulting `dashboard_final/output/` directory is self-contained and
can be zipped for distribution.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from shared_ui import CSS_MAIN  # noqa: E402
from pages import (  # noqa: E402
    home,
    methodology,
    how_it_works,
    experiment_31,
    experiment_32,
    robustness_caveats,
)


PAGES: list[tuple[str, str, "callable"]] = [
    # (filename, label, build_fn) — order matches the linear nav chain.
    ("home.html",                "Home",                 home.build),
    ("methodology.html",         "Methodology",          methodology.build),
    ("how_it_works.html",        "How it works",         how_it_works.build),
    ("experiment_31.html",       "Experiment §3.1",      experiment_31.build),
    ("experiment_32.html",       "Experiment §3.2",      experiment_32.build),
    ("robustness_caveats.html",  "Robustness & caveats", robustness_caveats.build),
]


def _copy_assets(out_dir: Path) -> list[tuple[str, int]]:
    out_assets = out_dir / "assets"
    out_assets.mkdir(parents=True, exist_ok=True)

    written: list[tuple[str, int]] = []

    # Plotly vendored.
    src_plotly = _HERE / "assets" / "plotly.min.js"
    if src_plotly.exists():
        dst_plotly = out_assets / "plotly.min.js"
        shutil.copyfile(src_plotly, dst_plotly)
        written.append(("assets/plotly.min.js", dst_plotly.stat().st_size))
    else:
        print(f"WARNING: vendored Plotly missing at {src_plotly}", file=sys.stderr)

    # Style sheet extracted from shared_ui.CSS_MAIN.
    style_path = out_assets / "style.css"
    style_path.write_text(CSS_MAIN.strip() + "\n", encoding="utf-8")
    written.append(("assets/style.css", style_path.stat().st_size))

    return written


def main() -> None:
    out_dir = _HERE / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[tuple[str, str, int]] = []
    for fname, label, build_fn in PAGES:
        try:
            html = build_fn()
        except Exception as e:
            print(f"FAILED to build {fname}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            raise
        path = out_dir / fname
        path.write_text(html, encoding="utf-8")
        written.append((fname, label, path.stat().st_size))

    asset_written = _copy_assets(out_dir)

    # Report
    print("=" * 70)
    print("dashboard_final — build report")
    print("=" * 70)
    print(f"Output directory: {out_dir}")
    print()
    print("Pages:")
    total = 0
    for fname, label, size in written:
        kb = size / 1024
        total += size
        print(f"  {fname:32s} {kb:>9.1f} KB   ({label})")
    print()
    print("Assets:")
    for fname, size in asset_written:
        kb = size / 1024
        total += size
        print(f"  {fname:32s} {kb:>9.1f} KB")
    print()
    print(f"Total bundle: {total / (1024*1024):.2f} MB")
    print()
    print("To distribute: zip the entire `output/` folder.")
    print("To view locally: open `output/home.html` in any browser.")


if __name__ == "__main__":
    main()
