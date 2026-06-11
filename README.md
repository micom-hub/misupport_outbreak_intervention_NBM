# Local Health Department Simulator (LHDSim)
*A modular simulation framework for Local Health Department interventions against outbreaks on realistic contact networks*

**Author:** Cameron Hempton  
**Created:** 24/10/2025

---

## Overview
This model simulates disease outbreaks on a contact network to evaluate how Local health department resource allocation policies during outbreak investigation influence disease outcomes. 

### About
In an outbreak response, local health departments (LHDs) apply their resources to interrupt disease transmission chains. The size of these contact chains scales exponentially, making it intractable to reach out to all contacts in the case of a highly-contagious disease like measles.

### Applications
- **Policy Evaluation:** Compare contact-tracing and isolation strategies (e.g., "Trace-then-Isolate" vs. "Observe Only"), and their 
- **Resource Optimization:** Determine the most efficient use of limited LHD resources, and optimize "explore versus exploit", or multi-armed bandit dynamics for the trade-off between learning more about outbreak structure versus taking direct action to interrupt chains of transmission.
- **Sensitivity Analysis:** Determine which epidemiological and population parameters (transmission probability, vaccination uptake, contact structure) most significantly impact outbreak outcomes, and how this changes depending on LHD resource allocation. This may lend to understanding optimal LHD intervention policy varies based on epidemic and population characteristics, leading to the development of "best practices in X scenario".

---

## Environment Setup
To set up the environment using the provided YAML file:
```bash
conda env create -f environment.yml
conda activate lhdsim
```

*Note: For `FredFetch` functionality, ensure a compatible `chromedriver` is installed, or manually place synthetic population ZIP files in `./data/{County}.zip`.*

---

## Execution Guide

### Quick Start (Single Run)
To run a single simulation with default parameters and a single LHD policy:
```bash
python scripts/singledriver.py
```
This is useful for debugging or visualizing a outbreak trajectories for a single set of outbreak parameters.

### Running Policy Variants
To compare different LHD prioritization algorithms, use the [variantdriver](scripts/variantdriver.py) script.

# Experimental Workflow:

#### Step 1: Configure Policy Variants
Edit [policy_config](scripts/lhd/policy_config.py) to define the LHD policies you are looking to compare. Ensure that policy_name exists within [policy_catalog](scripts/lhd/policy_catalog.py)
```python
# Example in scripts/lhd/policy_config.py
POLICY_CONFIGURATION = PolicyConfig(variants=[
    PolicyVariant(name="observe_only", policy_name="observe_only"),
    PolicyVariant(name="trace_then_isolate", policy_name="trace_then_isolate"),
])
```
#### Step 2: Run  Variant Driver
```bash
python scripts/variantdriver.py
```
This script will iterate through the defined variants, running the model across the specified parameter space or stochastic replicates.

### Parameter Sweeps and Sensitivity Analysis
To perform a global sensitivity analysis using Latin Hypercube Sampling (LHS) and Partial Rank Correlation Coefficient (PRCC):

1.  **Configure Parameters:** Update [config](scripts/config.py) to set the base ranges for transmission, vaccination uptake, and jurisdiction settings.
2.  **Generate LHS:** Manually write an {LHS}.csv file with columns: Parameter (parameter name in config), Minimum (min of parameter range), Maximum (max of parameter range), Integer (0/1, whether values should be coerced to an integer), ensure that VariantDriver points to this LHS
3.  **Execute Pipeline:** Use the full pipeline shell script to run the full simulation, post-processing, and PRCC analysis:
    ```bash
    chmod +x scripts/parameter_sweep_pipeline.sh
    ./scripts/parameter_sweep_pipeline.sh --run-dir model_runs/'__name_of_your_experiment_here__' --all --baseline-policy observe_only
    ```

*Note: To monitor run progress, navigate to your designated model run directory, where runs will be generated in order numbered 1-n_samples*
---



## Modifying Model Configurations

### Epidemiological & Jurisdiction Settings
Edit `scripts/config.py` to change:
- **Contact Network Structure:** Adjust how individuals make contact during the model
- **Epi Parameters:** Modify 
- **Simulation Settings:** Change `num_reps`  or `max_timesteps`.

### LHD Resource Logic
LHD behavior is defined in `scripts/lhd/policy_catalog.py`. You can modify how the LHD prioritizes individuals by:
1. Adding a new function to the catalog.
2. Registering it in `policy_config.py`.
3. Overriding default LHD parameters (like `daily_call_capacity`) within a `PolicyVariant`.

---

## Analytic Pipeline (MATLAB)
The model uses a MATLAB post-processing suite located in `scripts/PostRunProcessing/matlab_files/`:
- **`lhsPrccFromCsv.m`**: Conducts PRCC analysis on exported CSV results.
- **`performPrccWithCorrections.m`**: Applies Bonferroni and Benjamini-Hochberg (BHFDR) corrections to account for multiple testing.
- **`plotPRCC.m`**: Generates temporal sensitivity plots, showing how parameter importance shifts over the duration of an outbreak.


For this to work, ensure MATLAB is installed along with the statistics and machine learning toolkit, and is available in your system path.

---

