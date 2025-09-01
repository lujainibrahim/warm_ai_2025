import statsmodels.formula.api as smf
import statsmodels.api as sm
from analysis_utilities import (
    load_and_preprocess_all_data, 
    build_formula, 
    save_results_to_file,
    calculate_detailed_marginal_effects_with_tests
)

def main():    
    INCLUDE_LENGTH = False
    
    # Load and preprocess all data using shared function
    full_df = load_and_preprocess_all_data(include_length=INCLUDE_LENGTH)
    if full_df is None:
        return

    # --- MODEL 3: DETAILED AMENDMENT INTERACTION ANALYSIS ---
    print("\n" + "="*80)
    print("Model 3: Detailed Interpersonal Context Interaction Analysis")
    print("="*80)
    
    # Filter data: no user belief (prompt_type='original')
    df_model3 = full_df[full_df['prompt_type'] == 'original'].copy()
    
    print(f"Data for Model 3: {len(df_model3)} rows")

    if len(df_model3) > 0:
        base_formula_parts = [
            "is_incorrect ~ is_finetuned",
            "C(amendment_type_detailed, Treatment(reference='unmodified'))",
            "C(dataset, Treatment(reference='disinfo'))",
            "is_finetuned:C(amendment_type_detailed, Treatment(reference='unmodified'))",
            "C(model)"
        ]
        formula_mod_3 = build_formula(base_formula_parts, include_length=INCLUDE_LENGTH)
        
        try:
            model_mod_3 = smf.glm(
                formula=formula_mod_3,
                data=df_model3,
                family=sm.families.Binomial()
            ).fit()
            print("\nDetailed Amendment Interaction Model converged successfully!")
            print(model_mod_3.summary())
            
            print("\n" + "="*60)
            print("Marginal Effects with Statistical Tests:")
            print("="*60)
            
            detailed_results = calculate_detailed_marginal_effects_with_tests(model_mod_3, df_model3)
            
            for amend_type, stats in detailed_results.items():
                print(f"\n{amend_type.upper()}:")
                print(f"  Coefficient: {stats['coefficient']:.4f}")
                print(f"  Standard Error: {stats['standard_error']:.4f}")
                print(f"  P-value: {stats['p_value']:.4f}")
                
                # Calculate confidence interval
                coeff = stats['coefficient']
                se = stats['standard_error']
                ci_lower = coeff - 1.96 * se
                ci_upper = coeff + 1.96 * se
                print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                
                if 'interaction_coefficient' in stats:
                    print(f"  Interaction Coefficient: {stats['interaction_coefficient']:.4f}")

            output_filename = "model_3_detailed_interpersonal_context_interaction"
            if INCLUDE_LENGTH:
                output_filename += "_with_length"
            output_filename += ".txt"
            
            data_info = {
                "Data filter": "prompt_type == 'original'",
                "Total observations": f"{len(df_model3):,}",
                "Number of model fixed effects": df_model3['model'].nunique(),
                "Number of detailed amendment types": df_model3['amendment_type_detailed'].nunique()
            }
            
            save_results_to_file(
                filename=output_filename,
                title="MODEL 3: DETAILED INTERPERSONAL CONTEXT INTERACTION ANALYSIS",
                model_results=model_mod_3,
                data_info=data_info,
                marginal_effects_dict=detailed_results,
                include_length=INCLUDE_LENGTH
            )

            print(f"Results saved to {output_filename}")

        except Exception as e:
            print(f"Error in model 3: {e}")
            
    else:
        print("No data available for model 3 after filtering.")

if __name__ == "__main__":
    main()
