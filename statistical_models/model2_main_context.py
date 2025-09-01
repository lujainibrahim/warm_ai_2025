import statsmodels.formula.api as smf
import statsmodels.api as sm
from analysis_utilities import (
    load_and_preprocess_all_data, 
    build_formula, 
    save_results_to_file,
    calculate_marginal_effects_with_tests,
    calculate_conditional_marginal_effects
)


def main():
    INCLUDE_LENGTH = False
    
    full_df = load_and_preprocess_all_data(include_length=INCLUDE_LENGTH)
    if full_df is None:
        return

    # --- MODEL 2: AMENDMENT INTERACTION ANALYSIS ---
    print("\n" + "="*80)
    print("Model 2: Interpersonal Context Type Interaction Analysis")
    print("="*80)
    
    # Filter data: exclude user belief prompts (use only original prompts)
    df_model2 = full_df[full_df['prompt_type'] == 'original'].copy()
    
    print(f"Data for Model 2: {len(df_model2)} rows")

    if len(df_model2) > 0:
        base_formula_parts = [
            "is_incorrect ~ is_finetuned",
            "C(amendment_type, Treatment(reference='unmodified'))",
            "C(dataset, Treatment(reference='disinfo'))",
            "is_finetuned:C(amendment_type, Treatment(reference='unmodified'))",
            "C(model)"
        ]
        formula_mod_2 = build_formula(base_formula_parts, include_length=INCLUDE_LENGTH)
        
        try:
            model_mod_2 = smf.glm(
                formula=formula_mod_2,
                data=df_model2,
                family=sm.families.Binomial()
            ).fit()
            print(model_mod_2.summary())
            
            # Calculate marginal effects using counterfactual prediction
            counterfactual_effects = calculate_conditional_marginal_effects(
                model_mod_2, df_model2, 'amendment_type'
            )
            print(f"\Marginal Effects:")
            for var, effect in counterfactual_effects.items():
                print(f"  {var}: {effect:.4f} ({effect*100:.2f} pp)")
            
            # Calculate marginal effects using linear combinations of coefficients
            print("\n" + "="*60)
            print("Marginal Effects with Statistical Tests:")
            print("="*60)
            
            coefficient_effects = calculate_marginal_effects_with_tests(model_mod_2, df_model2)
            
            for context, stats in coefficient_effects.items():
                print(f"\n{context.upper()}:")
                print(f"  Marginal Effect: {stats['marginal_effect']:.4f} ({stats['marginal_effect']*100:.2f} pp)")
                print(f"  Standard Error: {stats['standard_error']:.4f}")
                print(f"  P-value: {stats['p_value']:.4f}")
                
                coeff = stats['marginal_effect']
                se = stats['standard_error']
                ci_lower = coeff - 1.96 * se
                ci_upper = coeff + 1.96 * se
                print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

            # Save results to file
            output_filename = "model_2_interpersonal_context_interaction"
            if INCLUDE_LENGTH:
                output_filename += "_with_length"
            output_filename += ".txt"
            
            combined_marginal_effects = {}
            for context, stats in coefficient_effects.items():
                combined_marginal_effects[f"{context} (with tests)"] = stats

            for var, effect in counterfactual_effects.items():
                combined_marginal_effects[f"{var} (counterfactual)"] = effect
            
            data_info = {
                "Data filter": "prompt_type == 'original'",
                "Total observations": f"{len(df_model2):,}",
                "Number of model fixed effects": df_model2['model'].nunique()
            }
            
            save_results_to_file(
                filename=output_filename,
                title="MODEL 2: INTERPERSONAL CONTEXT INTERACTION ANALYSIS",
                model_results=model_mod_2,
                data_info=data_info,
                marginal_effects_dict=combined_marginal_effects,
                include_length=INCLUDE_LENGTH
            )

            print(f"Results saved to {output_filename}")

        except Exception as e:
            print(f"Error in model 2: {e}")
            
    else:
        print("No data available for model 2 after filtering.")

if __name__ == "__main__":
    main()