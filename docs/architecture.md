# Architecture

This repo follows a **hybrid public layout**:

- reusable modules are promoted into `src/sia/core`
- legacy experiment code is grouped by use case under `use_cases/`
- only curated notebooks are kept
- generated results stay out of git and should be rebuilt from scripts

This preserves runnable provenance while avoiding a direct copy of the historical repo layout.
