import statsmodels.formula.api as smf
import statsmodels.api as sm
from analysis_utilities import (
    load_and_preprocess_all_data, 
    build_formula, 
    save_results_to_file,
    calculate_basic_marginal_effects,
    calculate_dataset_marginal_effects
)

def calculate_all_marginal_effects(model, data):
    results = {}
    # Fine-tuning effect
    results['Fine-tuning'] = calculate_basic_marginal_effects(model, data, 'is_finetuned')
    
    # Dataset effects (relative to disinfo)
    dataset_effects = calculate_dataset_marginal_effects(model, data)
    results.update(dataset_effects)
    
    return results

def main():
    
    INCLUDE_LENGTH = False
    
    full_df = load_and_preprocess_all_data(include_length=INCLUDE_LENGTH)
    if full_df is None:
        return

    # --- MODEL 1: MAIN EFFECTS LOGISTIC REGRESSION ---
    print("\n" + "="*80)
    print("Model 1: Main Effects")
    print("="*80)
    
    # Filter data: no interpersonal context (amendment_type='unmodified') and no user belief (prompt_type='original')
    df_model1 = full_df[(full_df['amendment_type'] == 'unmodified') & (full_df['prompt_type'] == 'original')].copy()
    
    print(f"Data for Model 1: {len(df_model1)} rows")

    if len(df_model1) > 0:
        base_formula_parts = [
            "is_incorrect ~ is_finetuned",
            "C(dataset, Treatment(reference='disinfo'))",
            "C(model)"
        ]
        formula_mod_1_logit = build_formula(base_formula_parts, include_length=INCLUDE_LENGTH)

        try:
            model_mod_1_logit = smf.glm(
                formula=formula_mod_1_logit,
                data=df_model1,
                family=sm.families.Binomial()
            ).fit()
            print(model_mod_1_logit.summary())

            # Calculate and save marginal effects for the logistic model
            marginal_effects_mod_1 = calculate_all_marginal_effects(model_mod_1_logit, df_model1)
            print(f"\nMarginal effects (in percentage points):")
            for var, effect in marginal_effects_mod_1.items():
                print(f"  {var}: {effect:.4f} ({effect*100:.2f} pp)")
            
            # Generate output filename and save results using shared function
            output_filename = "model_1_main_effects_logit"
            if INCLUDE_LENGTH:
                output_filename += "_with_length"
            output_filename += ".txt"
            
            data_info = {
                "Data filter": "amendment_type == 'unmodified' AND prompt_type == 'original'",
                "Total observations": f"{len(df_model1):,}",
                "Number of model fixed effects": df_model1['model'].nunique()
            }
            
            save_results_to_file(
                filename=output_filename,
                title="MODEL 1: MAIN EFFECTS",
                model_results=model_mod_1_logit,
                data_info=data_info,
                marginal_effects_dict=marginal_effects_mod_1,
                include_length=INCLUDE_LENGTH
            )

            print(f"Results saved to {output_filename}")

        except Exception as e:
            print(f"Error in model 1: {e}")
    else:
        print("No data available for model 1 after filtering.")

if __name__ == "__main__":
    main()