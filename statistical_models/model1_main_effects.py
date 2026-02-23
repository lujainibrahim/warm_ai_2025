import statsmodels.formula.api as smf
import statsmodels.api as sm
from analysis_utilities import (
    load_and_preprocess_all_data,
    build_formula,
    save_results_to_file,
    calculate_basic_marginal_effects,
    calculate_dataset_marginal_effects,
    calculate_response_length_marginal_effect,
)


def calculate_all_marginal_effects(model, data, include_length=False):
    results = {}
    results['Fine-tuning'] = calculate_basic_marginal_effects(model, data, 'is_finetuned')

    if include_length and 'response_length' in data.columns:
        results['Response length (+100 chars)'] = calculate_response_length_marginal_effect(model, data, delta=50)

    dataset_effects = calculate_dataset_marginal_effects(model, data)
    results.update(dataset_effects)
    return results


def run_model1(full_df, include_length):
    print("\n" + "=" * 80)
    label = "with" if include_length else "without"
    print(f"Model 1: Main Effects — {label} response length")
    print("=" * 80)

    df_model1 = full_df[
        (full_df['amendment_type'] == 'unmodified') &
        (full_df['prompt_type'] == 'original')
    ].copy()

    print(f"Data for Model 1: {len(df_model1)} rows")
    if len(df_model1) == 0:
        print("No data available for Model 1 after filtering.")
        return

    base_parts = [
        "is_incorrect ~ C(is_finetuned, Treatment(reference=0))",
        "C(dataset, Treatment(reference='disinfo'))",
        "C(model)",
    ]
    formula = build_formula(base_parts, include_length=include_length)

    try:
        result = smf.glm(
            formula=formula,
            data=df_model1,
            family=sm.families.Binomial(),
        ).fit()
        print(result.summary())

        marginal = calculate_all_marginal_effects(result, df_model1, include_length=include_length)
        print("\nMarginal Effects (percentage points):")
        for var, effect in marginal.items():
            print(f"  {var}: {effect:.4f} ({effect*100:.2f} pp)")

        out = "model_1_main_effects_no_context_logit"
        if include_length:
            out += "_with_length"
        out += ".txt"

        save_results_to_file(
            filename=out,
            title="MODEL 1: MAIN EFFECTS",
            model_results=result,
            data_info={
                "Data filter": "amendment_type == 'unmodified' AND prompt_type == 'original'",
                "Total observations": f"{len(df_model1):,}",
                "Number of model fixed effects": df_model1['model'].nunique(),
            },
            marginal_effects_dict=marginal,
            include_length=include_length,
        )
        print(f"Results saved to {out}")

    except Exception as e:
        print(f"Error in Model 1: {e}")


def main():
    full_df = load_and_preprocess_all_data()
    if full_df is None:
        return

    run_model1(full_df, include_length=False)
    run_model1(full_df, include_length=True)


if __name__ == "__main__":
    main()
