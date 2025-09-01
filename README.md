# Warm and empathetic AI systems are less reliable and more sycophantic

## Overview

This repository contains the analysis and visualization code for the submission ``Warm and empathetic AI systems are less reliable and more sycophantic."

The analysis pipeline includes:
- **Model 1**: Main effects analysis (baseline warmth fine-tuning effects without interpersonal context)
- **Model 2**: Interpersonal context interaction analysis (grouped amendment types)
- **Model 3**: Detailed interpersonal context analysis (specific amendment types)
- **Model 4**: Sycophancy analysis (user belief interaction effects)

## Repository Structure

```
warm_ai_2025/
├── statistical_models/           # Core statistical analysis code
│   ├── analysis_utilities.py     # Shared data processing and analysis functions
│   ├── model_configs_sample.py   # Configuration for sample data paths
│   ├── model1_main_effects.py    # Main effects logistic regression
│   ├── model2_main_context.py    # Context interaction analysis
│   ├── model3_detailed_context.py # Detailed context breakdown
│   └── model4_sycophancy.py      # Sycophancy/user belief analysis
├── sample_data/                  # Sample datasets for testing
│   ├── llama_70b/               # Llama-3 70B model outputs
│   │   ├── original/            # Original model outputs
│   │   └── warm/                # Warm model outputs
│   └── qwen_32b/                # Qwen-32B model outputs
│       ├── original/            # Original outputs
│       └── warm/                # Warm model outputs
├── summary_data/                 # Aggregated results and significance tests
├── figures/                      # Generated plots and visualizations
├── requirements.txt              # Full package dependencies                  
```

## Setup

All requirements can be found in `requirements.txt`. If you use conda, create a new environment and install the required dependencies:

```bash
conda create -n warm-ai python=3.11
conda activate warm-ai
git clone <repository-url>
cd warm_ai_2025
pip install -r requirements.txt
```

Similarly, if you use virtualenv, create a new environment and install the required dependencies:

```bash
python -m venv warm-ai
source warm-ai/bin/activate  # On Windows: warm-ai\Scripts\activate
git clone <repository-url>
cd warm_ai_2025
pip install -r requirements.txt
```

The setup should only take a few moments.

## Usage

### Data Configuration

Before running the statistical models, configure your data paths in `statistical_models/model_configs_sample.py`. The configuration expects CSV files with the following structure:

- **Original model outputs**: Original model responses to prompts
- **Warm model outputs**: Warm fine-tuned model responses to the same prompts

Each CSV must contain at least the following columns (additional meta data columns are allowed):
- `prompt_template`: Type of prompt used (e.g., 'original', 'incorrect')
- `amendment_type`: Interpersonal context modification (e.g., 'unmodified', 'relation:close', 'stake:high')
- `evaluation`: Response correctness ('CORRECT', 'INCORRECT')

### Running Statistical Models

Navigate to the statistical models directory:
```bash
cd statistical_models
```

**Model 1 - Main Effects Analysis:**
```bash
python model1_main_effects.py
```
Analyzes baseline warmth fine-tuning effects using only unmodified prompts and original prompt types. Generates `model_1_main_effects_logit.txt`.

**Model 2 - Interpersonal Context Interaction Analysis:**
```bash
python model2_main_context.py
```
Analyzes how interpersonal context types (emotion, relation, stake) interact with warmth fine-tuning effects. Generates `model_2_interpersonal_context_interaction.txt`.

**Model 3 - Detailed Interpersonal Context Analysis:**
```bash
python model3_detailed_context.py
```
Provides granular analysis of specific interpersonal context modifications. Generates `model_3_detailed_interpersonal_context_interaction.txt`.

**Model 4 - Sycophancy Analysis:**
```bash
python model4_sycophancy.py
```
Analyzes how user belief prompts (i.e., testing sycophancy) interact with warmth fine-tuning effects. Compares original prompts vs. user opinion prompts. Generates `model_4_sycophancy_analysis.txt`.

### Model Arguments and Configuration

Each model can be configured by modifying the `INCLUDE_LENGTH` variable in the respective Python files:
- `INCLUDE_LENGTH = False`: Standard analysis (default)
- `INCLUDE_LENGTH = True`: Includes model response length as a covariate

### Understanding the Output

Each model generate both **console output** and a **text file** with the detailed regression results, coefficients, and marginal effects.

**Expected Runtime:**
- **Sample data (316 observations)**: < 30 seconds per model on a standard desktop

### Sample Data

The repository includes sample data for testing:
- **2 models**: Llama-70B and Qwen-32B
- **4 datasets**: Disinfo, MedQA, TriviaQA, TruthfulQA
- **~20 observations per file**: Sufficient for testing model compilation

### Generating Figures

The `figures.ipynb` notebook contains code for generating all visualizations and figures (except for the summary methods figure 1) used in the paper. The notebook reads aggregated data from the `summary_data/` directory, which contains error scores per model, per dataset, per amendment type, and other summary statistics needed for visualization. 

**Note:** The aggregated summary data is produced by running `summary_data/significance_tests.py` on the full dataset (currently not included in this repository). 

