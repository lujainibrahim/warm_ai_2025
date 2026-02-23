import statsmodels.formula.api as smf
import statsmodels.api as sm
from analysis_utilities import (
    load_and_preprocess_all_data,
    build_formula,
    save_results_to_file,
    calculate_detailed_marginal_effects_with_tests,
)

INCLUDE_LENGTH = False


def main():
    full_df = load_and_preprocess_all_data()
    if full_df is None:
        return

    print("\n" + "=" * 80)
    print("Model 3: Detailed Interpersonal Context Interaction Analysis")
    print("=" * 80)

    df_model3 = full_df[full_df['prompt_type'] == 'original'].copy()
    print(f"Data for Model 3: {len(df_model3)} rows")

    if len(df_model3) == 0:
        print("No data available for Model 3 after filtering.")
        return

    print(f"\nDetailed amendment types:")
    for amend_type in sorted(df_model3['amendment_type_detailed'].unique()):
        count = len(df_model3[df_model3['amendment_type_detailed'] == amend_type])
        print(f"  {amend_type} (n={count:,})")

    base_parts = [
        "is_incorrect ~ C(is_finetuned, Treatment(reference=0))",
        "C(amendment_type_detailed, Treatment(reference='unmodified'))",
        "C(dataset, Treatment(reference='disinfo'))",
        "C(is_finetuned, Treatment(reference=0)):C(amendment_type_detailed, Treatment(reference='unmodified'))",
        "C(model)",
    ]
    formula = build_formula(base_parts, include_length=INCLUDE_LENGTH)

    try:
        result = smf.glm(
            formula=formula,
            data=df_model3,
            family=sm.families.Binomial(),
        ).fit()
        print(result.summary())

        detailed = calculate_detailed_marginal_effects_with_tests(result, df_model3)

        print("\nMarginal Effects with Statistical Tests:")
        for amend_type, st in detailed.items():
            me = st.get('marginal_effect')
            print(f"\n  {amend_type.upper()}:")
            if me is not None:
                print(f"    Marginal Effect: {me:.4f} ({me*100:.2f} pp)")
            print(f"    Coefficient: {st['coefficient']:.4f}")
            print(f"    SE: {st['standard_error']:.4f},  p={st['p_value']:.4f}")
            ci_lo = st['coefficient'] - 1.96 * st['standard_error']
            ci_hi = st['coefficient'] + 1.96 * st['standard_error']
            print(f"    95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
            if 'interaction_coefficient' in st:
                print(f"    Interaction Coefficient: {st['interaction_coefficient']:.4f}")

        out = "model_3_detailed_amendment_interaction_results"
        if INCLUDE_LENGTH:
            out += "_with_length"
        out += ".txt"

        save_results_to_file(
            filename=out,
            title="MODEL 3: DETAILED INTERPERSONAL CONTEXT INTERACTION ANALYSIS",
            model_results=result,
            data_info={
                "Data filter": "prompt_type == 'original'",
                "Total observations": f"{len(df_model3):,}",
                "Number of model fixed effects": df_model3['model'].nunique(),
                "Number of detailed amendment types": df_model3['amendment_type_detailed'].nunique(),
            },
            marginal_effects_dict=detailed,
            include_length=INCLUDE_LENGTH,
        )
        print(f"Results saved to {out}")

    except Exception as e:
        print(f"Error in Model 3: {e}")


if __name__ == "__main__":
    main()
