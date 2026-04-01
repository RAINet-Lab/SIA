# ABR use case

This folder contains the Adaptive Bitrate Streaming variants used in the SIA work:

- `vanilla/`: baseline ABR agent
- `lumos/`: forecast-augmented ABR variant
- `xatu/`: alternative forecast-augmented ABR variant
- `sia_refiner/`: SIA-guided refinement workflow

Run `python scripts/bootstrap_public_data.py --dataset abr` from the repository root before using these variants. The bootstrap script downloads the public Norway mobility traces, converts them into the legacy two-column format, and wires the compatibility directories expected by the migrated code.
