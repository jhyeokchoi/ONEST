# ONEST: Optimized Novel Exchange Saturation Transfer

**A Web-Based Platform for the Rapid and Robust Analysis of Protein Excited States through CEST Spectroscopy**

## Overview

ONEST (Optimized Novel Exchange Saturation Transfer) is a user-friendly tool designed to automate and accelerate the analysis of Chemical Exchange Saturation Transfer (CEST) NMR spectroscopy data. It facilitates the characterization of "invisible" protein excited states—transient, low-population conformations that are critical for biological function but undetectable by conventional structural methods.

Unlike traditional methods that rely on computationally intensive numerical integration, ONEST utilizes a simultaneous multi-field fitting algorithm leveraging exact analytical solutions for two-state exchange (the Baldwin model). This approach significantly enhances both the speed and robustness of the analysis.

## Key Features

*   **Rapid Analysis:** Utilizes the Baldwin model for exact analytical solutions, avoiding slow numerical integration.
*   **Robust Fitting:** Simultaneous multi-field fitting algorithm for reliable characterization of exchange parameters.
*   **User-Friendly:** Designed to be accessible for researchers analyzing CEST data.
*   **Web-Based & Local Execution:** Can be run as a local web server or via command-line scripts.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jhyeokchoi/ONEST/
    cd ONEST
    ```

2.  **Install dependencies:**
    Run the following command to install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    > **Tip:** It is recommended to use a virtual environment to avoid conflicts with other projects.
    > ```bash
    > python3 -m venv venv
    > source venv/bin/activate  # On Windows: venv\Scripts\activate
    > ```

## Usage

### 1. Running the Web Server
To start the web-based interface:
```bash
python server_run.py
```
Access the interface at `http://127.0.0.1:5000`.

### 2. Command Line Execution
You can run the analysis scripts directly using a configuration file:
```bash
python run.py config.json
```

> **Tip (For Matrix Method & Benchmarking):**
> When using the **Matrix method** or running benchmarks, it is recommended to force single-core execution to avoid overhead or ensure consistent timing.
> ```bash
> export OMP_NUM_THREADS=1
> python run.py config.json
> ```

### 3. Benchmarking
To benchmark the performance of the model fitting:
```bash
python benchmark.py config.json [profile]
```
- `config.json`: Path to your configuration file.
- `profile`: (Optional) Add this argument to enable cProfile and generate a `benchmark_profile.prof` file.

## Directory Structure

*   **`run.py`**: Main execution script for standard fitting via command line.
*   **`server_run.py`**: Flask-based web server for the graphical user interface.
*   **`mcrun.py`**: Script for Monte Carlo simulations (supports multiprocessing).
*   **`benchmark.py`**: Tool for benchmarking model performance.
*   **`estmodel.py`**: Core library containing calculation engines (Baldwin, Matrix, NoEx) and model logic.
*   **`example/`**: Contains example datasets (`syn10.txt`, `syn100.txt`) for testing.
*   **`requirements.txt`**: Python dependencies.

## Citation

If you use ONEST in your research, please cite:

> Choi, J., Lee, SY., Han, K., Carneiro, M. G., Ryu, KS., & Lee, D. "ONEST: A Web-Based Platform for the Rapid and Robust Analysis of Protein Excited States through CEST Spectroscopy." (in preparation)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
