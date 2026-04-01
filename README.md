# SIA

Public-facing repository for **SIA: Symbolic Interpretability for Anticipatory Deep Reinforcement Learning in Network Control**.

## Repository layout

- `src/sia/core/`: reusable SIA building blocks migrated from the original research repo.
- `use_cases/abr/`: ABR agents and SIA-aware refinement variants.
- `use_cases/mimo/`: curated MIMO analysis notebooks and environment notes.
- `use_cases/ran_slicing/`: curated RAN-slicing scripts/notebooks and public result artifacts.
- `notebooks/`: small curated entrypoint notebooks matching the paper workflow.
- `scripts/`: smoke tests and dataset bootstrap helpers.
- `docs/`: architecture and migration notes.

## Status

This repo is being extracted from the historical `mobicom-2024` research repo into a cleaner public structure. Generated artifacts and large datasets are intentionally excluded from git.

## Dataset access policy

### ABR

`python scripts/bootstrap_public_data.py --dataset abr` downloads public Norway mobility traces from the Pensieve/Simula lineage, converts them into the legacy two-column format used by the migrated code, and prepares compatibility links for the ABR variants.

### RAN slicing

This public repo ships only the scripts, curated notebooks, and exported result artifacts for the RAN-slicing use case. The underlying dataset is confidential and is not redistributed here. For access, contact the Northeastern University coauthors listed in the SIA paper: Leonardo Bonati, Salvatore D'Oro, Michele Polese, and Tommaso Melodia (`{l.bonati, s.doro, m.polese, t.melodia}@northeastern.edu`).

### Massive MIMO

This public repo ships the analysis code/notebooks and cites the original environment paper: *A Deep Reinforcement Learning-Based Resource Scheduler for Massive MIMO Networks* (Qing An, Santiago Segarra, Chris Dick, Ashutosh Sabharwal, and Rahman Doost-Mohammady, IEEE TMLCN 2023). The training traces are not redistributed here. For trace access, contact the original paper authors, especially the Rice University group behind that environment.

## Quick start

1. Create a Python environment and install the package dependencies from `pyproject.toml`.
2. Run `python scripts/bootstrap_public_data.py --dataset abr` to prepare the public ABR traces.
3. Run `python scripts/smoke_imports.py` to validate the basic package imports.
