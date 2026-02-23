import statsmodels.formula.api as smf
import statsmodels.api as sm
from analysis_utilities import (
    load_and_preprocess_all_data,
    build_formula,
    save_results_to_file,
    calculate_sycophancy_marginal_effects_with_tests,
)

INCLUDE_LENGTH = False


def main():
    full_df = load_and_preprocess_all_data()
    if full_df is None:
        return

    print("\n" + "=" * 80)
    print("Model 4: Sycophancy (User Belief) Interaction Analysis")
    print("=" * 80)

    df_model4 = full_df[full_df['prompt_type'].isin(['original', 'user_opinion'])].copy()
    print(f"Data for Model 4: {len(df_model4)} rows")

    if len(df_model4) == 0:
        print("No data available for Model 4 after filtering.")
        return

    print(f"\nPrompt type distribution:")
    for pt in df_model4['prompt_type'].unique():
        count = len(df_model4[df_model4['prompt_type'] == pt])
        print(f"  {pt}: {count:,} observations")

    base_parts = [
        "is_incorrect ~ C(is_finetuned, Treatment(reference=0))",
        "C(amendment_type, Treatment(reference='unmodified'))",
        "C(dataset, Treatment(reference='disinfo'))",
        "C(is_finetuned, Treatment(reference=0)):C(prompt_type, Treatment(reference='original'))",
        "C(model)",
    ]
    formula = build_formula(base_parts, include_length=INCLUDE_LENGTH)

    try:
        result = smf.glm(
            formula=formula,
            data=df_model4,
            family=sm.families.Binomial(),
        ).fit()
        print(result.summary())

        syc = calculate_sycophancy_marginal_effects_with_tests(result, df_model4)

        print("\nMarginal Effects with Statistical Tests:")
        for prompt_type, st in syc.items():
            me = st.get('marginal_effect')
            print(f"\n  {prompt_type.upper()} PROMPTS:")
            if me is not None:
                print(f"    Marginal Effect: {me:.4f} ({me*100:.2f} pp)")
            print(f"    Coefficient: {st['coefficient']:.4f}")
            print(f"    SE: {st['standard_error']:.4f},  p={st['p_value']:.4f}")
            ci_lo = st['coefficient'] - 1.96 * st['standard_error']
            ci_hi = st['coefficient'] + 1.96 * st['standard_error']
            print(f"    95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

        out = "model_4_sycophancy_analysis_results"
        if INCLUDE_LENGTH:
            out += "_with_length"
        out += ".txt"

        save_results_to_file(
            filename=out,
            title="MODEL 4: SYCOPHANCY (USER BELIEF) INTERACTION ANALYSIS",
            model_results=result,
            data_info={
                "Data filter": "prompt_type in ['original', 'user_opinion']",
                "Total observations": f"{len(df_model4):,}",
                "Number of model fixed effects": df_model4['model'].nunique(),
                "Prompt types": str(list(df_model4['prompt_type'].unique())),
            },
            marginal_effects_dict=syc,
            include_length=INCLUDE_LENGTH,
        )
        print(f"Results saved to {out}")

    except Exception as e:
        print(f"Error in Model 4: {e}")


if __name__ == "__main__":
    main()
