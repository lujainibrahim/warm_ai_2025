import pandas as pd
import numpy as np
import os
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import fdrcorrection
from itertools import groupby
from analysis_utilities import filter_dataframe
from model_configs import ALL_CONFIGS


all_configs = ALL_CONFIGS


def filter_and_prepare_df(df, dataset):
    df_filtered, stats = filter_dataframe(df, dataset)

    # Unify column names for prompt_type
    if 'prompt_template' in df_filtered.columns and 'prompt_type' not in df_filtered.columns:
        df_filtered = df_filtered.rename(columns={'prompt_template': 'prompt_type'})

    df_filtered['is_correct'] = df_filtered['evaluation'].str.contains(r'\bCORRECT\b', case=False, na=False, regex=True).astype(int)
    
    # Rename the merge key for merging base and ft data
    keyword_map = {
        'truthfulqa': 'content_key', 'disinfo': 'prompt_disinfo',
        'trivia': 'content_key', 'medqa': 'original_question'
    }
    join_key = keyword_map.get(dataset)
    if join_key and join_key in df_filtered.columns:
        df_filtered['__merge_key'] = (
            df_filtered[join_key].astype(str).str.strip() + '||' +
            df_filtered['amendment_type'].astype(str).str.strip() + '||' +
            df_filtered['prompt_type'].astype(str).str.strip()
        )
        df_prepared = df_filtered.rename(columns={'__merge_key': 'merge_key'})
        return df_prepared[['merge_key', 'amendment_type', 'prompt_type', 'is_correct']], stats
    
    stats['final'] = 0 
    return pd.DataFrame(), stats

def run_mcnemar_tests(merged_df, category_col, baseline_label, ft_label):
    results_data = []
    for category, group in merged_df.groupby(category_col):
        if len(group) == 0:
            continue

        base_correct = group['is_correct_base']
        ft_correct = group['is_correct_ft']

        # McNemar's table components
        both_correct = ((base_correct == 1) & (ft_correct == 1)).sum()
        base_only = ((base_correct == 1) & (ft_correct == 0)).sum()  # b
        ft_only = ((base_correct == 0) & (ft_correct == 1)).sum()   # c  
        both_wrong = ((base_correct == 0) & (ft_correct == 0)).sum()
        
        table = [[both_correct, base_only],
                 [ft_only, both_wrong]]
        
        pct_base = base_correct.mean() * 100
        pct_ft = ft_correct.mean() * 100
        
        # Effect size calculations
        raw_diff = pct_ft - pct_base  # Raw percentage point difference
        
        # Cohen's g for McNemar's test
        cohens_g = np.nan
        if (base_only + ft_only) > 0:
            cohens_g = abs(base_only - ft_only) / np.sqrt(base_only + ft_only)
        
        # Odds Ratio for McNemar's test
        odds_ratio = np.nan
        if ft_only > 0:
            odds_ratio = base_only / ft_only
        elif base_only > 0:
            odds_ratio = np.inf  
        
        # Effect size interpretation
        if not np.isnan(cohens_g):
            if cohens_g < 0.2:
                g_interp = "Small"
            elif cohens_g < 0.5:
                g_interp = "Medium"
            else:
                g_interp = "Large"
        else:
            g_interp = "N/A"

        p_value = np.nan
        if (base_only + ft_only) > 0: 
            try:
                result = mcnemar(table, exact=True)
                p_value = result.pvalue
            except Exception as e:
                pass 

        results_data.append({
            'Category': category,
            f'{baseline_label} (%)': pct_base,
            f'{ft_label} (%)': pct_ft,
            'Raw Diff (pp)': raw_diff,
            'Cohen\'s g': cohens_g,
            'g Effect Size': g_interp,
            'Odds Ratio': odds_ratio,
            'p-value-raw': p_value
        })
        
    if not results_data:
        return pd.DataFrame()

    results_df = pd.DataFrame(results_data)

    # Apply FDR correction on non-NaN p-values
    valid_p_values_idx = results_df['p-value-raw'].notna()
    if valid_p_values_idx.any():
        p_values = results_df.loc[valid_p_values_idx, 'p-value-raw']
        rejected, pvals_corrected = fdrcorrection(p_values, alpha=0.05, method='indep')
        
        results_df.loc[valid_p_values_idx, 'p-value (FDR corrected)'] = pvals_corrected
        results_df.loc[valid_p_values_idx, 'Significant (FDR)'] = np.where(rejected, 'Yes', 'No')

    # Add original significance for comparison
    results_df['Significant (p < 0.05)'] = (results_df['p-value-raw'] < 0.05).map({True: 'Yes', False: 'No'})


    results_df[f'{baseline_label} (%)'] = results_df[f'{baseline_label} (%)'].apply(lambda x: f"{x:.2f}")
    results_df[f'{ft_label} (%)'] = results_df[f'{ft_label} (%)'].apply(lambda x: f"{x:.2f}")
    results_df['Raw Diff (pp)'] = results_df['Raw Diff (pp)'].apply(lambda x: f"{x:.2f}")
    results_df['Cohen\'s g'] = results_df['Cohen\'s g'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    results_df['Odds Ratio'] = results_df['Odds Ratio'].apply(lambda x: f"{x:.2f}" if pd.notna(x) and not np.isinf(x) else ("∞" if np.isinf(x) else "N/A"))
    results_df['p-value'] = results_df['p-value-raw'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    final_cols = ['Category', f'{baseline_label} (%)', f'{ft_label} (%)', 'Raw Diff (pp)', 
                  'Cohen\'s g', 'g Effect Size', 'Odds Ratio', 'p-value', 'Significant (p < 0.05)']
    
    if 'p-value (FDR corrected)' in results_df.columns:
        results_df['p-value (FDR corrected)'] = results_df['p-value (FDR corrected)'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        final_cols.extend(['p-value (FDR corrected)', 'Significant (FDR)'])
        
    return results_df[final_cols].sort_values(by='Category')

def main():
    results_dir = "significance_results"
    os.makedirs(results_dir, exist_ok=True)
    all_configs.sort(key=lambda x: x['model'])
    model_groups = {k: list(v) for k, v in groupby(all_configs, key=lambda x: x['model'])}

    all_sycophancy_results = []
    all_normal_results = []
    all_filtering_stats = []

    for model, configs in model_groups.items():
        print(f"\n{'='*60}\nRunning McNemar's Paired Tests for Model: {model.upper()}\n{'='*60}")

        for config in configs:
            dataset = config['dataset']
            print(f"\n--- Dataset: {dataset.upper()} ---")

            try:
                base_data = pd.read_csv(config['base_path'])
                ft_data = pd.read_csv(config['ft_path'])
            except FileNotFoundError as e:
                print(f"  Error loading data: {e}")
                continue

            base_filtered, base_stats = filter_and_prepare_df(base_data, dataset)
            ft_filtered, ft_stats = filter_and_prepare_df(ft_data, dataset)

            # Collect filtering statistics for CSV
            all_filtering_stats.append({
                'Model': model,
                'Dataset': dataset,
                'Model_Type': 'Base',
                'Initial_Rows': base_stats['initial'],
                'Refusals_Count': base_stats['refusals'],
                'Refusals_Treatment': 'Kept as CORRECT' if dataset == 'disinfo' else 'Filtered Out',
                'Final_Rows': base_stats['final']
            })
            
            all_filtering_stats.append({
                'Model': model,
                'Dataset': dataset,
                'Model_Type': 'Fine-Tuned',
                'Initial_Rows': ft_stats['initial'],
                'Refusals_Count': ft_stats['refusals'],
                'Refusals_Treatment': 'Kept as CORRECT' if dataset == 'disinfo' else 'Filtered Out',
                'Final_Rows': ft_stats['final']
            })

            print("\n  -- Sample Sizes --")
            print(f"    Base Model:")
            print(f"      Initial Rows:        {base_stats['initial']}")
            if dataset == 'disinfo':
                print(f"      Refusals (kept as 'CORRECT'): {base_stats['refusals']}")
            else:
                print(f"      Refusals Filtered Out:       {base_stats['refusals']}")
            print(f"      Final Rows for Analysis:     {base_stats['final']}")
            
            print(f"    Fine-Tuned Model:")
            print(f"      Initial Rows:        {ft_stats['initial']}")
            if dataset == 'disinfo':
                print(f"      Refusals (kept as 'CORRECT'): {ft_stats['refusals']}")
            else:
                print(f"      Refusals Filtered Out:       {ft_stats['refusals']}")
            print(f"      Final Rows for Analysis:     {ft_stats['final']}")

            if base_filtered.empty or ft_filtered.empty or 'merge_key' not in base_filtered.columns:
                print("  Could not prepare data for merging (missing merge key or empty). Skipping.")
                continue

            merged_df = pd.merge(
                base_filtered.rename(columns={'is_correct': 'is_correct_base'}),
                ft_filtered.rename(columns={'is_correct': 'is_correct_ft'}),
                on='merge_key',
                suffixes=('_base', '_ft')
            )

            if merged_df.empty:
                print("  No common prompts found after filtering. Skipping.")
                continue
                
            # --- Sycophancy Analysis ---
            syc_df = merged_df[merged_df['prompt_type_base'].str.contains('correct|incorrect', case=False, na=False, regex=True)].copy()
            if not syc_df.empty:
                print("\n  -- Significance Tests for Sycophancy Analysis --")
                syc_results_df = run_mcnemar_tests(syc_df, 'amendment_type_base', f"{model} Base", f"{model} FT")
                print(syc_results_df.to_string())
                
                if not syc_results_df.empty:
                    syc_results_df_export = syc_results_df.copy()
                    syc_results_df_export = syc_results_df_export.rename(columns={
                        f'{model} Base (%)': 'Base (%)',
                        f'{model} FT (%)': 'FT (%)'
                    })
                    syc_results_df_export['Model'] = model
                    syc_results_df_export['Dataset'] = dataset
                    syc_results_df_export['Test_Type'] = 'Sycophancy'
                    all_sycophancy_results.append(syc_results_df_export)
                    
                    csv_filename = f"significance_{model}_{dataset}_sycophancy.csv"
                    csv_path = os.path.join(results_dir, csv_filename)
                    syc_results_df.to_csv(csv_path, index=False)
                    print(f"  Saved sycophancy results to: {csv_path}")

            # --- Main Analysis ---
            normal_df = merged_df.copy()
            normal_df['amendment_type_clean'] = normal_df['amendment_type_base'].astype(str).str.strip()
            normal_df['prompt_type_clean'] = normal_df['prompt_type_base'].astype(str).str.strip()
            normal_df['template_group'] = normal_df['prompt_type_clean']
            mask = (normal_df['amendment_type_clean'] == 'unmodified') & (normal_df['prompt_type_clean'].isin(['original', 'original neutral']))
            normal_df.loc[mask, 'template_group'] = 'original_combined'
            
            normal_df_filtered = normal_df[
                ((normal_df['amendment_type_clean'] == 'unmodified') & (normal_df['template_group'] == 'original_combined')) |
                ((normal_df['amendment_type_clean'] != 'unmodified') & (normal_df['prompt_type_clean'].isin(['original', 'original neutral'])))
            ].copy()
            
            if not normal_df_filtered.empty:
                normal_df_filtered['category'] = normal_df_filtered['amendment_type_clean'] + "\n" + normal_df_filtered['template_group']
                print("\n  -- Significance Tests for Normal Categories --")
                normal_results_df = run_mcnemar_tests(normal_df_filtered, 'category', f"{model} Base", f"{model} FT")
                print(normal_results_df.to_string())
                
                if not normal_results_df.empty:
                    normal_results_df_export = normal_results_df.copy()
                    normal_results_df_export = normal_results_df_export.rename(columns={
                        f'{model} Base (%)': 'Base (%)',
                        f'{model} FT (%)': 'FT (%)'
                    })
                    normal_results_df_export['Model'] = model
                    normal_results_df_export['Dataset'] = dataset
                    normal_results_df_export['Test_Type'] = 'Normal'
                    all_normal_results.append(normal_results_df_export)
                    
                    csv_filename = f"significance_{model}_{dataset}_normal.csv"
                    csv_path = os.path.join(results_dir, csv_filename)
                    normal_results_df.to_csv(csv_path, index=False)
                    print(f"  Saved main results to: {csv_path}")

    if all_sycophancy_results:
        combined_syc_df = pd.concat(all_sycophancy_results, ignore_index=True)
        combined_syc_df.to_csv("significance_all_models_sycophancy.csv", index=False)
        print(f"\nSaved combined sycophancy results to: significance_all_models_sycophancy.csv")

    if all_normal_results:
        combined_normal_df = pd.concat(all_normal_results, ignore_index=True)
        combined_normal_df.to_csv("significance_all_models_normal.csv", index=False)
        print(f"Saved combined main results to: significance_all_models_normal.csv")

    # One master CSV with everything
    if all_sycophancy_results or all_normal_results:
        all_results = []
        if all_sycophancy_results:
            all_results.extend(all_sycophancy_results)
        if all_normal_results:
            all_results.extend(all_normal_results)
        
        if all_results:
            master_df = pd.concat(all_results, ignore_index=True)
            master_df.to_csv("significance_all_models_complete.csv", index=False)
            print(f"Saved master CSV with all results to: significance_all_models_complete.csv")

    # Save filtering (refusal) statistics
    if all_filtering_stats:
        filtering_df = pd.DataFrame(all_filtering_stats)
        filtering_df.to_csv("filtering_statistics_all_models.csv", index=False)
        print(f"Saved filtering statistics to: filtering_statistics_all_models.csv")

if __name__ == "__main__":
    main() 