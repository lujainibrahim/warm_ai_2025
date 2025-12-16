import statsmodels.formula.api as smf
import statsmodels.api as sm
from analysis_utilities import (
    load_and_preprocess_all_data, 
    build_formula, 
    save_results_to_file,
    calculate_sycophancy_marginal_effects_with_tests
)

def main():
    INCLUDE_LENGTH = False
    
    full_df = load_and_preprocess_all_data(include_length=INCLUDE_LENGTH)
    if full_df is None:
        return

    print("\n" + "="*80)
    print("Model 4: Sycophancy (User Belief) Interaction Analysis")
    print("="*80)
    
    df_model4 = full_df[full_df['prompt_type'].isin(['original', 'user_opinion'])].copy()
    
    print(f"Data for Model 4: {len(df_model4)} rows")

    if len(df_model4) > 0:
        base_formula_parts = [
            "is_incorrect ~ C(is_finetuned, Treatment(reference=0))",
            "C(prompt_type, Treatment(reference='original'))",
            "C(dataset, Treatment(reference='disinfo'))",
            "C(is_finetuned, Treatment(reference=0)):C(prompt_type, Treatment(reference='original'))",
            "C(model)"
        ]
        formula_mod_4 = build_formula(base_formula_parts, include_length=INCLUDE_LENGTH)
        
        try:
            model_mod_4 = smf.glm(
                formula=formula_mod_4,
                data=df_model4,
                family=sm.families.Binomial()
            ).fit()
            print(model_mod_4.summary())
            
            print("\n" + "="*60)
            print("Marginal Effects with Statistical Tests:")
            print("="*60)
            
            sycophancy_results = calculate_sycophancy_marginal_effects_with_tests(model_mod_4, df_model4)
            
            for prompt_type, stats in sycophancy_results.items():
                print(f"\n{prompt_type.upper()} PROMPTS:")
                print(f"  Marginal Effect: {stats['marginal_effect']:.4f} ({stats['marginal_effect']*100:.2f} pp)")
                print(f"  Standard Error: {stats['standard_error']:.4f}")
                print(f"  P-value: {stats['p_value']:.4f}")
                
                coeff = stats['marginal_effect']
                se = stats['standard_error']
                ci_lower = coeff - 1.96 * se
                ci_upper = coeff + 1.96 * se
                print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

            output_filename = "model_4_sycophancy_analysis"
            if INCLUDE_LENGTH:
                output_filename += "_with_length"
            output_filename += ".txt"
            
            data_info = {
                "Data filter": "prompt_type in ['original', 'user_opinion']",
                "Total observations": f"{len(df_model4):,}",
                "Number of model fixed effects": df_model4['model'].nunique(),
                "Prompt types": list(df_model4['prompt_type'].unique())
            }
            
            save_results_to_file(
                filename=output_filename,
                title="MODEL 4: SYCOPHANCY (USER BELIEF) INTERACTION ANALYSIS",
                model_results=model_mod_4,
                data_info=data_info,
                marginal_effects_dict=sycophancy_results,
                include_length=INCLUDE_LENGTH
            )

            print(f"Results saved to {output_filename}")

        except Exception as e:
            print(f"Error in model 4: {e}")
            
    else:
        print("No data available for model 4 after filtering.")

if __name__ == "__main__":
    main()
