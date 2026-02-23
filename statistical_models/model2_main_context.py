import statsmodels.formula.api as smf
import statsmodels.api as sm
from analysis_utilities import (
    load_and_preprocess_all_data,
    build_formula,
    save_results_to_file,
    calculate_marginal_effects_with_tests,
    calculate_conditional_marginal_effects,
)

INCLUDE_LENGTH = False


def main():
    full_df = load_and_preprocess_all_data()
    if full_df is None:
        return

    print("\n" + "=" * 80)
    print("Model 2: Interpersonal Context Type Interaction Analysis")
    print("=" * 80)

    df_model2 = full_df[full_df['prompt_type'] == 'original'].copy()
    print(f"Data for Model 2: {len(df_model2)} rows")

    if len(df_model2) == 0:
        print("No data available for Model 2 after filtering.")
        return

    base_parts = [
        "is_incorrect ~ C(is_finetuned, Treatment(reference=0))",
        "C(amendment_type, Treatment(reference='unmodified'))",
        "C(dataset, Treatment(reference='disinfo'))",
        "C(is_finetuned, Treatment(reference=0)):C(amendment_type, Treatment(reference='unmodified'))",
        "C(model)",
    ]
    formula = build_formula(base_parts, include_length=INCLUDE_LENGTH)

    try:
        result = smf.glm(
            formula=formula,
            data=df_model2,
            family=sm.families.Binomial(),
        ).fit()
        print(result.summary())

        coeff_effects = calculate_marginal_effects_with_tests(result, df_model2)

        print("\nMarginal Effects with Statistical Tests:")
        for context, st in coeff_effects.items():
            me = st.get('marginal_effect')
            print(f"\n  {context.upper()}:")
            if me is not None:
                print(f"    Marginal Effect: {me:.4f} ({me*100:.2f} pp)")
            print(f"    Coefficient: {st['coefficient']:.4f}")
            print(f"    SE: {st['standard_error']:.4f},  p={st['p_value']:.4f}")
            ci_lo = st['coefficient'] - 1.96 * st['standard_error']
            ci_hi = st['coefficient'] + 1.96 * st['standard_error']
            print(f"    95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

        out = "model_2_amendment_interaction_no_user_belief"
        if INCLUDE_LENGTH:
            out += "_with_length"
        out += ".txt"

        save_results_to_file(
            filename=out,
            title="MODEL 2: INTERPERSONAL CONTEXT INTERACTION ANALYSIS",
            model_results=result,
            data_info={
                "Data filter": "prompt_type == 'original'",
                "Total observations": f"{len(df_model2):,}",
                "Number of model fixed effects": df_model2['model'].nunique(),
            },
            marginal_effects_dict=coeff_effects,
            include_length=INCLUDE_LENGTH,
        )
        print(f"Results saved to {out}")

    except Exception as e:
        print(f"Error in Model 2: {e}")


if __name__ == "__main__":
    main()
