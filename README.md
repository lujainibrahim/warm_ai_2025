# Training language models to be warm can undermine accuracy and increase sycophancy

## Overview

This repository contains the analysis and visualization code for the submission "Training language models to be warm can undermine accuracy and increase sycophancy"

The analysis pipeline includes:
- **Model 1**: Main effects analysis (baseline warmth fine-tuning effects without interpersonal context)
- **Model 2**: Interpersonal context interaction analysis (grouped amendment types)
- **Model 3**: Detailed interpersonal context analysis (specific amendment types)
- **Model 4**: Sycophancy analysis (user belief interaction effects)

## Repository Structure

```
├── statistical_models/              # Core statistical analysis code
│   ├── analysis_utilities.py        # Shared data processing, filtering, and marginal effects
│   ├── model_configs_sample.py      # Configuration for sample data paths
│   ├── model1_main_effects.py       # Model 1: Main effects logistic regression
│   ├── model2_main_context.py       # Model 2: Context interaction analysis
│   ├── model3_detailed_context.py   # Model 3: Detailed context breakdown
│   ├── model4_sycophancy.py         # Model 4: Sycophancy/user belief analysis
│   └── model_outputs/               # Results of model outputs
├── sample_data/                     # Sample datasets for testing
│   ├── llama_70b/                   # Llama-3 70B model outputs
│   │   ├── original/
│   │   └── warm/
│   └── qwen_32b/                    # Qwen-32B model outputs
│       ├── original/
│       └── warm/
├── source_data/                     # Source data for figures
│   ├── sociot_ft.csv                # SocioT fine-tuning warmth scores
│   ├── social_syco.csv              # Social sycophancy benchmark results
│   ├── results_warm_ft.csv          # Warm fine-tuning results
│   ├── results_warm_sysprompt.csv   # Warm system prompt results
│   ├── results_cold_ft.csv          # Cold fine-tuning results
│   └── general_benchmark.csv        # General benchmark results
├── summary_data/                    # Aggregated results and significance tests
├── eval_data/                       # Evaluation datasets (JSON)
│   ├── disinfo.json
│   ├── medqa.json
│   ├── trivia.json
│   └── truthfulqa.json
├── figures.ipynb                    # Notebook for generating all figures
└── requirements.txt                 # Python package dependencies
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
source warm-ai/bin/activate  
git clone <repository-url>
cd warm_ai_2025
pip install -r requirements.txt
```

The setup should only take a few moments.

## Usage

### Data Configuration

Before running the statistical models, configure your data paths in `statistical_models/model_configs_sample.py`. The configuration lists all 5 model families used in the paper (Llama-3 70B, Llama-3 8B, Mistral Small, Qwen-32B, GPT-4o) with paths pointing to the `sample_data/` directory. The sample data includes 2 of these models (Llama-3 70B and Qwen-32B) for testing.

Each CSV file must contain at least the following columns (additional metadata columns are allowed):
- `prompt_template`: Type of prompt used (e.g., 'original', 'incorrect')
- `amendment_type`: Interpersonal context modification (e.g., 'unmodified', 'relation:close', 'stake:high')
- `evaluation`: Response correctness ('CORRECT', 'INCORRECT')
- `output`: Model response text (used for refusal detection and response length calculation)

### Running Statistical Models

Navigate to the statistical models directory:
```bash
cd statistical_models
```

**Model 1 -- Main Effects Analysis:**
```bash
python model1_main_effects.py
```
Analyzes baseline warmth fine-tuning effects using only unmodified prompts and original prompt types. Set `INCLUDE_LENGTH = True` to include response length as a covariate, or `INCLUDE_LENGTH = False` to exclude it. Generates `model_1_main_effects_logit_with_length.txt` or `model_1_main_effects_logit.txt` accordingly.

**Model 2 -- Interpersonal Context Interaction Analysis:**
```bash
python model2_main_context.py
```
Analyzes how interpersonal context types (emotion, relation, stake) interact with warmth fine-tuning effects. Generates `model_2_interpersonal_context_interaction.txt`.

**Model 3 -- Detailed Interpersonal Context Analysis:**
```bash
python model3_detailed_context.py
```
Provides granular analysis of specific interpersonal context modifications (e.g., relation:close, stake:high). Generates `model_3_detailed_interpersonal_context_interaction.txt`.

**Model 4 -- Sycophancy Analysis:**
```bash
python model4_sycophancy.py
```
Analyzes how user belief prompts (i.e., testing sycophancy) interact with warmth fine-tuning effects. Compares original prompts vs. user opinion prompts. Generates `model_4_sycophancy_analysis.txt`.


### Model Arguments and Configuration

Each model can be configured by modifying the `INCLUDE_LENGTH` variable in the respective Python files:
- `INCLUDE_LENGTH = False`: Standard analysis (default for Models 2--4)
- `INCLUDE_LENGTH = True`: Includes model response length as a covariate (default for Model 1)

### Understanding the Output

Each model generates both **console output** and a **text file** with detailed regression results, coefficients, and marginal effects with statistical tests (p-values, confidence intervals). Pre-generated outputs from the full dataset are available in `statistical_models/model_outputs/`.

**Expected Runtime:**
- **Sample data (316 observations)**: < 30 seconds per model on a standard desktop

### Sample Data

The repository includes sample data for testing:
- **2 models**: Llama-3 70B and Qwen-32B
- **4 datasets**: Disinfo, MedQA, TriviaQA, TruthfulQA
- **~20 observations per file**: Sufficient for testing model compilation

### Generating Figures

The `figures.ipynb` notebook contains code for generating all visualizations and figures used in the paper. The notebook reads data from `source_data/` (for fine-tuning scores, sycophancy benchmarks, and general benchmarks) and `summary_data/` (for aggregated error scores per model, per dataset, per amendment type).

**Note:** The aggregated summary data is produced by running `summary_data/significance_tests.py` on the full dataset.
