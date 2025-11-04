KGroot (Bank dataset) — Setup and Run Guide

This repository contains a runnable pipeline to process the Bank telemetry dataset into graph data and train the KGroot model on it.

What you will do:
- Process raw telemetry under `Bank/` into graphs in `bank_from_bank/`
- Generate `labeled_data.json`
- Train KGroot and write logs and model artifacts


Prerequisites
- Python 3.8–3.10 recommended
- Git, PowerShell or a terminal
- Optional: CUDA-capable GPU + correct PyTorch build


Quickstart (Windows PowerShell)
1) Clone and enter the project
   ```bash
   git clone https://github.com/<your-username>/KGroot.git
   cd KGroot
   ```

2) Create and activate a virtual environment
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3) Install dependencies
   - If you have a GPU, install a matching PyTorch first from the official selector, e.g. CPU-only:
     ```bash
     pip install torch==1.4.0 torchvision==0.5.0 --index-url https://download.pytorch.org/whl/cpu
     ```
   - Then install the project requirements (legacy, broad list; some packages may already be satisfied):
     ```bash
     pip install -r requirements.txt
     ```

4) Verify dataset layout
   Ensure the Bank telemetry is present at:
   - `Bank/query.csv`
   - `Bank/record.csv`
   - `Bank/telemetry/YYYY_MM_DD/{log,metric,trace}/...csv`

   The repository already includes a sample `Bank/` structure. Replace or add dates as needed.

5) Run the full pipeline (process data + train)
   ```bash
   python run_bank_kgroot.py
   ```


What the script does
- Processes a few dates from `Bank/telemetry/` into pickles under `bank_from_bank/pickle_data/`
- Creates `bank_from_bank/labeled_data.json`
- Trains KGroot using `graph_sim_dej_X.py` with settings from `config_graph_sim.ini`
- Writes TensorBoard logs to `runs/` and saves model files under `data/graph_sim_model_parameters/...`


Configuration
- Training/config parameters live in `config_graph_sim.ini`:
  - `[data]`: dataset_dir (`bank_from_bank`), train/valid/test file names, batch size, resplit flags
  - `[model]`: `input_dim` (must match 100), `support` (3), `gcn_hidden_dim`, `linear_hidden_dim`, `max_node_num`
  - `[train]`: `LR`, `NB_EPOCH`, weight decay, comment

- Data processing knobs (edit `bank_data_processor.py` if needed):
  - `process_all_dates(max_dates=3, sample_size=1000)` controls how many dates and how many rows per file are sampled for speed.


Outputs
- Processed graphs: `bank_from_bank/pickle_data/*.pkl`
- Labels file: `bank_from_bank/labeled_data.json`
- Training logs: `runs/<timestamp>_*`
- Saved model + copied config: `data/graph_sim_model_parameters/<dataset>/<timestamp_host+comment>/`


Visualize training with TensorBoard
```bash
tensorboard --logdir runs
```
Open the shown URL in your browser to inspect metrics and histograms.


Troubleshooting
- GPU device index:
  - `model.py` selects `cuda:1` if CUDA is available. If you have a single GPU (usually `cuda:0`) or CPU only, change the line:
    ```python
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    ```
    to
    ```python
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ```
    or force CPU by setting it to `"cpu"`.

- Dependency versions:
  - The `requirements.txt` is broad and legacy. If an install fails, install PyTorch first (matching your CUDA), then retry `pip install -r requirements.txt`. You can also install a minimal set: `numpy`, `pandas`, `scipy`, `networkx`, `tensorboardX`, plus `torch`/`torchvision`.

- Data size/performance:
  - Increase `max_dates` and/or `sample_size` in `bank_data_processor.py` once the pipeline runs end-to-end.


Git hygiene
- Secrets and large artifacts are ignored via `.gitignore` (includes `.env`, `.venv/`, caches, pickles, CSVs, zips, `runs/`). If `.env` was ever committed, untrack it once:
  ```bash
  git rm --cached .env
  git commit -m "Stop tracking env file"
  ```


Run just the data processor (optional)
```bash
python -m bank_data_processor
```
This will populate `bank_from_bank/` and exit.


Project entry points
- `run_bank_kgroot.py`: end-to-end runner (process + train)
- `graph_sim_dej_X.py`: training loop and evaluation (reads `config_graph_sim.ini`)
- `bank_data_processor.py`: converts raw telemetry to graphs and labels
