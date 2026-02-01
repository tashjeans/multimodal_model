## Multimodal TCR Model

A multimodal learning pipeline for modeling T-cell receptor (TCR) and peptide binding, integrating sequence and structure information using Boltz and a parameter efficient fine tuned protein language model. The multimodal model will be fine tuned with non-contrastive learning, and will aim to address the discrepancy between TCR and peptide promiscuity. 

---

## Purpose

This project aims to build a reproducible, high-accuracy model for predicting TCR–peptide interactions using multiple input modalities (e.g., TCRα/β sequences, peptide sequences, and structural parameters). It is designed for academic publication and real-world interpretability.

---

## Repo Structure

multimodal_model/
├── data/                 # Raw and processed data files
│   ├── raw/              # Original unmodified data (read-only)
│   └── processed/        # Cleaned/converted data (e.g., YAML, FASTA)
├── scripts/              # Python scripts (training, inference, conversion)
│   ├── preprocess/       # Data cleaning and format converters
│   └── train/            # Training and evaluation routines
├── models/               # Saved model checkpoints, architecture code
├── notebooks/            # Jupyter notebooks for EDA, experiments, etc.
├── utils/                # Helper functions (e.g., metrics, loaders)
├── config/               # YAML/JSON configs for experiments
├── tests/                # Unit tests (if applicable)
├── README.md             # Overview, usage, and instructions
├── requirements.txt      # Python dependencies (pip)
└── env.yml               # Conda environment file (alternative to pip)


## 🗂 Project Structure


## Notes
To reactivate environment with dependencies and boltz:
run: 'conda activate tcr-multimodal'

Database for HLA molecules:
https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/



