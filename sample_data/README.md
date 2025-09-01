# Sample Data for Journal Submission

This directory contains a representative sample of the original research data used in our study.

## Data Source
- **Original location**: /Users/lujainibrahim/Downloads/RB Files
- **Sample size**: 20 rows per file
- **Sampling method**: Random sampling with fixed seed (reproducible)
- **Purpose**: Demonstrate code functionality for peer review

## Important Notes

⚠️ **This is a small sample of the full dataset used in the paper.**
- Results from this sample will differ from those reported in the paper
- The sample demonstrates code functionality, not research findings
- For replication of published results, the full dataset would be required

## Structure

```
sample_data/
├── llama_70b/
│   ├── original/          # Base model outputs
│   │   ├── disinfo_original_outputs.csv
│   │   ├── medqa_original_outputs.csv
│   │   ├── trivia_original_outputs.csv
│   │   └── truthfulqa_original_outputs.csv
│   └── warm/              # Fine-tuned model outputs
│       ├── disinfo_warm_outputs.csv
│       ├── medqa_warm_outputs.csv
│       ├── trivia_warm_outputs.csv
│       └── truthfulqa_warm_outputs.csv
├── llama_8b/
│   └── [same structure]
├── mistral_small/
│   └── [same structure]
├── qwen_32b/
│   └── [same structure]
└── gpt4o/
    └── [same structure]
```

## Data Columns

The CSV files contain the original research data columns:
- `evaluation`: Model output evaluation ('correct'/'incorrect')
- `amendment_type`: Context manipulation categories
- `prompt_type`: Prompt variant types  
- `response_length`: Response length metrics
- Various merge keys for paired statistical tests
- Other metadata columns from the original study

## Usage

1. Update `model_configs.py` to point to this sample data:
   ```python
   'base_path': 'sample_data/llama_70b/original/disinfo_original_outputs.csv'
   'ft_path': 'sample_data/llama_70b/warm/disinfo_warm_outputs.csv'
   ```

2. Run analysis scripts:
   ```bash
   python model1_basic.py
   python model2_interpersonal_context.py  
   python model3_detailed_context.py
   python model4_sycophancy.py
   python significance_tests.py
   ```

## Data Privacy

This sample contains only the minimal data necessary to demonstrate the computational methods. No personally identifiable information or proprietary content is included.
