# SIA: Symbolic Interpretability for Anticipatory Deep Reinforcement Learning in Network Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) ![DOI](https://img.shields.io/badge/DOI-TBD-lightgrey.svg)

This repository contains the code and resources for the research paper accepted at **IEEE INFOCOM 2026**:

> **SIA: Symbolic Interpretability for Anticipatory Deep Reinforcement Learning in Network Control**  
> *MohammadErfan Jabbari, Abhishek Duttagupta, Claudio Fiandrino, Leonardo Bonati, Salvatore D'Oro, Michele Polese, Marco Fiore, and Tommaso Melodia*  
>  
> IMDEA Networks Institute, Spain  
> Universidad Carlos III de Madrid, Spain  
> Northeastern University, Boston, USA  
>  
> Email: `{name.surname}@networks.imdea.org`, `{l.bonati, s.doro, m.polese, t.melodia}@northeastern.edu`

## Abstract
Deep Reinforcement Learning (DRL) promises adaptive control for future mobile networks, but conventional agents remain reactive: they act on past and current measurements and cannot leverage short-term forecasts of exogenous KPIs such as bandwidth. Augmenting agents with predictions can overcome this temporal myopia, yet uptake in networking is scarce because forecast-aware agents behave as closed boxes; operators cannot tell whether predictions truly guide decisions or justify the added complexity. We propose **SIA**, the first interpreter that exposes in real time how forecast-augmented DRL agents operate. SIA combines symbolic AI abstractions with per-KPI knowledge graphs to generate explanations and introduces a new influence score metric. SIA achieves sub-millisecond runtime, more than **200×** faster than existing XAI methods. We evaluate SIA on three diverse networking use cases, uncovering hidden issues including temporal misalignment in forecast integration and reward-design biases that trigger counterproductive policies. These insights enable targeted fixes: a redesigned agent improves average video bitrate by **9%**, and SIA's online Action-Refinement module improves RAN-slicing reward by **25%** without retraining. By making anticipatory DRL transparent and tunable, SIA lowers the barrier to proactive control in next-generation mobile networks.

## Citation
If you find this work useful, please cite our paper:

```bibtex
@inproceedings{sia2026,
  title={SIA: Symbolic Interpretability for Anticipatory Deep Reinforcement Learning in Network Control},
  author={Jabbari, MohammadErfan and Duttagupta, Abhishek and Fiandrino, Claudio and Bonati, Leonardo and D'Oro, Salvatore and Polese, Michele and Fiore, Marco and Melodia, Tommaso},
  booktitle={IEEE INFOCOM 2026},
  year={2026},
  note={DOI to be added after publication. Available online: https://github.com/RAINet-Lab/SIA}
}
```

This repository contains the public SIA implementation and artifacts for three representative networking workloads: adaptive bitrate streaming (ABR), massive MIMO scheduling, and RAN slicing. It includes the migrated SIA core package, forecasting utilities, curated notebooks, and use-case-specific analysis code used to reproduce the public parts of the paper workflow.

## Key Contributions
- Introduces **SIA**, a symbolic interpretability framework for anticipatory DRL agents in mobile networks.
- Provides a reusable **core package** under `src/sia/` for symbolic abstraction, decision-graph construction, and forecasting support.
- Validates SIA across **three networking domains**: ABR, massive MIMO, and RAN slicing.
- Includes a public **ABR bootstrap pipeline** that reconstructs the trace layout expected by the migrated agents from publicly accessible network traces.
- Documents the **data-access boundary** for the confidential or third-party datasets used in the MIMO and RAN-slicing workflows.

## Repository Structure
```text
SIA/
├── src/sia/                  # Reusable SIA package
│   ├── core/                 # Symbolization, knowledge-graph, and decision logic
│   └── forecasting/          # Forecasting layers and support code
├── use_cases/
│   ├── abr/                  # ABR agents: vanilla, Lumos, Xatu, and SIA refiner
│   ├── mimo/                 # Massive-MIMO notebooks and analysis entrypoints
│   └── ran_slicing/          # RAN-slicing notebooks and public result workflow
├── notebooks/                # Curated top-level entrypoint notebooks
├── scripts/                  # Bootstrap helpers and smoke tests
├── docs/                     # Architecture and migration notes
├── data/                     # Lightweight placeholders for raw/processed data roots
└── tests/                    # Lightweight repository checks
```

## Getting Started
1. **Clone the repository**
   ```bash
   git clone git@github.com:RAINet-Lab/SIA.git
   cd SIA
   ```

2. **Create an environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

3. **Run the lightweight validation checks**
   ```bash
   python scripts/smoke_imports.py
   python tests/test_repo_layout.py
   ```

4. **Bootstrap the public ABR traces**
   ```bash
   python scripts/bootstrap_public_data.py --dataset abr
   ```

## Data Access
- **ABR**: Publicly bootstrapable through `scripts/bootstrap_public_data.py`, which downloads Norway mobility traces and reconstructs the layout expected by the migrated ABR code.
- **RAN slicing**: The dataset is confidential. This repository ships the scripts, notebooks, and public results only. For dataset access, contact the Northeastern University coauthors via the paper email block: `{l.bonati, s.doro, m.polese, t.melodia}@northeastern.edu`.
- **Massive MIMO**: This repository ships the analysis code and notebooks, but not the original training traces. The environment is based on *A Deep Reinforcement Learning-Based Resource Scheduler for Massive MIMO Networks* (IEEE TMLCN 2023). For trace access, contact the original paper authors, especially the Rice University group behind that scheduler environment.

## License
This project is licensed under the MIT License. See the [`LICENSE`](./LICENSE) file for details.

## Contact
For questions, please open an issue on this repository or contact the paper authors.
