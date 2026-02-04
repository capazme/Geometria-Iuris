"""
archive_run.py — Archivia gli output di una run per il versioning.

Sposta tutti i file di output (results.json, visualization.html, plots/)
in una cartella etichettata sotto runs/, insieme a una copia della config.
Lo spazio output/ viene svuotato per la prossima run.

Usage:
    .venv/bin/python archive_run.py                    # etichetta auto (timestamp)
    .venv/bin/python archive_run.py "baseline_e5_bge"  # etichetta manuale
    .venv/bin/python archive_run.py --list              # elenca run archiviate
    .venv/bin/python archive_run.py --dry-run           # mostra cosa farebbe

Struttura risultante:
    runs/
    ├── 2026-02-04_baseline_e5_bge/
    │   ├── config.yaml
    │   ├── results.json
    │   ├── visualization.html
    │   ├── plots/
    │   │   ├── rsa_rdm_heatmaps.png
    │   │   ├── axes_projection_comparison.png
    │   │   ├── clustering_dendrograms.png
    │   │   └── nda_analysis.png
    │   └── MANIFEST.txt
    └── 2026-02-04_15h42_unnamed/
        └── ...
"""

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_FILE = PROJECT_ROOT / "config.yaml"

# ─── Simboli per i log ──────────────────────────────────────────────
_OK = "\033[32m✓\033[0m"   # verde
_WARN = "\033[33m!\033[0m" # giallo
_ERR = "\033[31m✗\033[0m"  # rosso
_INFO = "\033[36m→\033[0m" # ciano
_MOVE = "\033[34m⤷\033[0m" # blu


def _log(symbol: str, msg: str) -> None:
    print(f"  {symbol} {msg}")


def _sanitize_label(label: str) -> str:
    """Rende il label sicuro per un nome di directory."""
    label = label.strip().lower()
    label = re.sub(r"[^\w\-]", "_", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label[:60]


def _collect_output_files() -> list[Path]:
    """Elenca tutti i file nella directory di output."""
    if not OUTPUT_DIR.exists():
        return []
    files = []
    for p in OUTPUT_DIR.rglob("*"):
        if p.is_file():
            files.append(p)
    return sorted(files)


def _read_metadata(results_path: Path) -> dict:
    """Estrae i metadati da results.json se presente."""
    if not results_path.exists():
        return {}
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("metadata", {})
    except (json.JSONDecodeError, KeyError):
        return {}


def _read_summary(results_path: Path) -> dict:
    """Estrae un riepilogo dei risultati sperimentali da results.json."""
    if not results_path.exists():
        return {}
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        experiments = data.get("experiments", {})
        summary = {}
        rsa = experiments.get("1_rsa", {})
        if rsa:
            summary["rsa"] = f"r={rsa.get('spearman_r', '?'):.4f}, p={rsa.get('p_value', '?'):.4f}"
        gw = experiments.get("2_gromov_wasserstein", {})
        if gw:
            summary["gw"] = f"dist={gw.get('distance', '?'):.6f}, p={gw.get('p_value', '?'):.4f}"
        nda = experiments.get("5_nda", {})
        nda_a = nda.get("part_a_neighborhoods", {})
        if nda_a:
            summary["nda"] = f"J={nda_a.get('mean_jaccard', '?'):.4f}, p={nda_a.get('p_value', '?'):.4f}"
        return summary
    except Exception:
        return {}


def _build_run_name(label: str | None) -> str:
    """Costruisce il nome della cartella: YYYY-MM-DD_HHhMM_label."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%Hh%M")

    if label:
        clean = _sanitize_label(label)
        return f"{date_str}_{clean}"
    else:
        return f"{date_str}_unnamed"


def _write_manifest(dest: Path, files_moved: list[tuple[str, int]],
                    metadata: dict, label: str | None) -> None:
    """Scrive un file MANIFEST.txt con i dettagli dell'archiviazione."""
    lines = [
        f"CLS Pipeline — Run Archive",
        f"{'=' * 50}",
        f"Archiviata: {datetime.now().isoformat()}",
        f"Etichetta:  {label or '(nessuna)'}",
        f"",
    ]

    if metadata:
        lines.append("Metadati della run:")
        lines.append(f"  Pipeline version: {metadata.get('pipeline_version', '?')}")
        lines.append(f"  Timestamp run:    {metadata.get('timestamp_utc', '?')}")
        lines.append(f"  Config hash:      {metadata.get('config_hash', '?')[:16]}")
        lines.append(f"  Data hash:        {metadata.get('input_data_hash', '?')[:16]}")
        lines.append(f"  Random seed:      {metadata.get('random_seed', '?')}")
        lines.append(f"  Device:           {metadata.get('device', '?')}")
        models = metadata.get("models", {})
        for name, info in models.items():
            lines.append(f"  Modello {name}:    {info.get('name', '?')}")
        lines.append("")

    lines.append(f"File archiviati ({len(files_moved)}):")
    total_size = 0
    for rel_path, size in files_moved:
        size_kb = size / 1024
        total_size += size
        lines.append(f"  {rel_path:<45} {size_kb:>8.1f} KB")

    lines.append(f"")
    lines.append(f"Dimensione totale: {total_size / 1024:.1f} KB ({total_size / (1024*1024):.2f} MB)")

    manifest_path = dest / "MANIFEST.txt"
    manifest_path.write_text("\n".join(lines), encoding="utf-8")


def list_runs() -> int:
    """Elenca tutte le run archiviate."""
    if not RUNS_DIR.exists():
        _log(_WARN, "Nessuna run archiviata (la cartella runs/ non esiste).")
        return 0

    dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()])
    if not dirs:
        _log(_WARN, "Nessuna run archiviata.")
        return 0

    print(f"\n  Run archiviate ({len(dirs)}):")
    print(f"  {'─' * 65}")

    for d in dirs:
        files = list(d.rglob("*"))
        n_files = sum(1 for f in files if f.is_file())
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        results = d / "results.json"
        meta_info = ""
        if results.exists():
            meta = _read_metadata(results)
            if meta.get("timestamp_utc"):
                meta_info = f"  run: {meta['timestamp_utc'][:19]}"

        print(f"    {d.name:<40} {n_files:>3} file, {total_size/1024:>7.0f} KB{meta_info}")

    print(f"  {'─' * 65}")
    return 0


def archive_run(label: str | None = None, dry_run: bool = False) -> int:
    """Archivia la run corrente."""

    print(f"\n{'=' * 60}")
    print(f"  CLS Pipeline — Archiviazione Run")
    print(f"{'=' * 60}")

    # ── Passo 1: scansione output/ ──
    _log(_INFO, "Scansione della cartella output/ ...")
    output_files = _collect_output_files()
    if not output_files:
        _log(_WARN, "Niente da archiviare: la cartella output/ è vuota.")
        return 0

    files_to_move = []
    for f in output_files:
        rel = f.relative_to(OUTPUT_DIR)
        size = f.stat().st_size
        files_to_move.append((str(rel), size))
        _log(_INFO, f"Trovato: {str(rel):<42} ({size/1024:.1f} KB)")

    total = sum(s for _, s in files_to_move)
    _log(_OK, f"Trovati {len(files_to_move)} file, dimensione totale: {total/1024:.1f} KB")

    # ── Passo 2: lettura metadati da results.json ──
    results_path = OUTPUT_DIR / "results.json"
    if results_path.exists():
        _log(_INFO, "Lettura metadati da results.json ...")
        metadata = _read_metadata(results_path)
        if metadata:
            _log(_OK, f"Pipeline v{metadata.get('pipeline_version', '?')}, "
                       f"seed={metadata.get('random_seed', '?')}, "
                       f"device={metadata.get('device', '?')}")
            models = metadata.get("models", {})
            for name, info in models.items():
                _log(_INFO, f"Modello {name}: {info.get('name', '?')} (dim={info.get('dimension', '?')})")
        else:
            _log(_WARN, "results.json trovato ma senza metadati validi")

        summary = _read_summary(results_path)
        if summary:
            _log(_INFO, "Riepilogo risultati sperimentali:")
            for exp, val in summary.items():
                _log(_INFO, f"  {exp.upper()}: {val}")
    else:
        metadata = {}
        _log(_WARN, "results.json non trovato — archiviazione senza metadati")

    # ── Passo 3: costruzione nome destinazione ──
    run_name = _build_run_name(label)
    dest = RUNS_DIR / run_name
    _log(_INFO, f"Nome archivio: {run_name}")
    _log(_INFO, f"Percorso completo: {dest}")

    # ── Passo 4: verifica config.yaml ──
    has_config = CONFIG_FILE.exists()
    if has_config:
        config_size = CONFIG_FILE.stat().st_size
        _log(_OK, f"config.yaml trovato ({config_size/1024:.1f} KB) — verrà copiato")
    else:
        _log(_WARN, "config.yaml non trovato — l'archivio non includerà la configurazione")

    # ── Riepilogo pre-archiviazione ──
    print(f"\n  {'─' * 56}")
    print(f"  Riepilogo:")
    print(f"    Destinazione:  runs/{run_name}/")
    print(f"    File output:   {len(files_to_move)}")
    print(f"    Config:        {'sì' if has_config else 'no'}")
    print(f"    Dimensione:    {total/1024:.1f} KB ({total/(1024*1024):.2f} MB)")
    print(f"  {'─' * 56}")

    if dry_run:
        _log(_WARN, "[DRY RUN] Nessun file spostato. Rimuovi --dry-run per eseguire.")
        return 0

    # ── Passo 5: creazione directory destinazione ──
    if dest.exists():
        _log(_ERR, f"La cartella {dest.name} esiste già!")
        _log(_ERR, "Usa un'etichetta diversa o rinomina la cartella esistente.")
        return 1

    _log(_INFO, f"Creazione cartella runs/{run_name}/ ...")
    dest.mkdir(parents=True)
    _log(_OK, "Cartella creata")

    # ── Passo 6: spostamento file ──
    _log(_INFO, "Spostamento file di output ...")
    moved_count = 0
    for f in output_files:
        rel = f.relative_to(OUTPUT_DIR)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(target))
        moved_count += 1
        _log(_MOVE, f"{str(rel):<42} → runs/{run_name}/{rel}")

    _log(_OK, f"{moved_count} file spostati")

    # ── Passo 7: copia config ──
    if has_config:
        _log(_INFO, "Copia config.yaml nell'archivio ...")
        shutil.copy2(str(CONFIG_FILE), str(dest / "config.yaml"))
        _log(_OK, "config.yaml copiato")

    # ── Passo 8: scrittura manifest ──
    _log(_INFO, "Scrittura MANIFEST.txt ...")
    _write_manifest(dest, files_to_move, metadata, label)
    _log(_OK, "MANIFEST.txt scritto")

    # ── Passo 9: pulizia output/ ──
    _log(_INFO, "Pulizia cartella output/ ...")
    removed_dirs = 0
    for d in sorted(OUTPUT_DIR.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            removed_dirs += 1
    if removed_dirs:
        _log(_OK, f"Rimosse {removed_dirs} sotto-cartelle vuote")

    _log(_INFO, "Ricreazione struttura output/plots/ ...")
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
    _log(_OK, "output/plots/ ricreata (vuota)")

    # ── Fatto ──
    print(f"\n{'═' * 60}")
    _log(_OK, f"Archiviazione completata: runs/{run_name}/")
    _log(_OK, f"output/ svuotata e pronta per la prossima run.")

    # Mostra contenuto finale dell'archivio
    archived_files = sorted(dest.rglob("*"))
    archived_count = sum(1 for f in archived_files if f.is_file())
    archived_size = sum(f.stat().st_size for f in archived_files if f.is_file())
    _log(_INFO, f"Archivio finale: {archived_count} file, {archived_size/1024:.1f} KB")
    print(f"{'═' * 60}\n")

    return 0


def main() -> int:
    args = sys.argv[1:]

    if "--list" in args or "-l" in args:
        return list_runs()

    dry_run = "--dry-run" in args or "-n" in args
    args = [a for a in args if a not in ("--list", "-l", "--dry-run", "-n")]

    label = args[0] if args else None

    return archive_run(label=label, dry_run=dry_run)


if __name__ == "__main__":
    sys.exit(main())
