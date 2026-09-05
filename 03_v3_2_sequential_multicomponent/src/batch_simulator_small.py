import numpy as np
import pandas as pd

from sequential_assembly import (
    create_initial_state,
    deviation_offset,
    deviation_tilt,
    deviation_bend,
    deviation_waviness,
    deviation_twist,
    correction_profile,
    update_assembly_state,
    calculate_quality_metrics,
)


RNG = np.random.default_rng(42)

N_ASSEMBLIES = 10
N_COMPONENTS = 5


def generate_component(component_index):
    """
    Generate one controlled synthetic component deviation.
    Different component positions use different deviation families.
    """

    if component_index == 1:
        value = RNG.normal(0.0, 0.35)
        return deviation_offset(value), "offset", value

    elif component_index == 2:
        value = RNG.normal(0.0, 0.025)
        return deviation_tilt(value), "tilt", value

    elif component_index == 3:
        value = abs(RNG.normal(0.20, 0.08))
        return deviation_bend(value), "bend", value

    elif component_index == 4:
        value = abs(RNG.normal(0.10, 0.04))
        return deviation_waviness(value, waves=3), "waviness", value

    elif component_index == 5:
        value = RNG.normal(0.0, 0.15)
        return deviation_twist(value), "twist", value

    else:
        raise ValueError("Unsupported component index")


def simple_manual_correction(component_index, deviation_value):
    """
    Temporary rule-based correction baseline.

    This is NOT the final intelligent strategy.
    It is only used to verify batch simulation behaviour.
    """

    if component_index == 1:
        return correction_profile(
            z_adj_mm=-deviation_value
        )

    elif component_index == 2:
        return correction_profile(
            theta_adj_deg=-deviation_value
        )

    elif component_index == 3:
        return correction_profile(
            z_adj_mm=-0.5 * deviation_value
        )

    elif component_index == 4:
        return correction_profile(
            locator_offset_mm=-0.5 * deviation_value
        )

    elif component_index == 5:
        return correction_profile(
            theta_adj_deg=-0.03 * deviation_value
        )

    return correction_profile()


records = []


for assembly_id in range(1, N_ASSEMBLIES + 1):

    # --------------------------------------------------------
    # NO-CORRECTION STATE
    # --------------------------------------------------------

    state_nominal = create_initial_state()

    # --------------------------------------------------------
    # CORRECTED STATE
    # --------------------------------------------------------

    state_corrected = create_initial_state()


    for component_index in range(1, N_COMPONENTS + 1):

        deviation_profile, scenario, deviation_value = generate_component(
            component_index
        )

        # --------------------------------------------
        # NOMINAL / NO CORRECTION
        # --------------------------------------------

        state_nominal = update_assembly_state(
            previous_state=state_nominal,
            component_deviation=deviation_profile,
        )

        nominal_metrics = calculate_quality_metrics(
            state_nominal
        )

        # --------------------------------------------
        # SIMPLE CORRECTED BASELINE
        # --------------------------------------------

        correction = simple_manual_correction(
            component_index,
            deviation_value,
        )

        state_corrected = update_assembly_state(
            previous_state=state_corrected,
            component_deviation=deviation_profile,
            correction=correction,
        )

        corrected_metrics = calculate_quality_metrics(
            state_corrected
        )

        # --------------------------------------------
        # STORE COMPONENT-LEVEL RESULT
        # --------------------------------------------

        records.append(
            {
                "assembly_id": assembly_id,
                "component_index": component_index,
                "scenario": scenario,
                "deviation_value": deviation_value,

                "nominal_mean_gap":
                    nominal_metrics["mean_gap"],

                "nominal_max_gap":
                    nominal_metrics["max_gap"],

                "nominal_parallelism":
                    nominal_metrics["parallelism_error"],

                "nominal_quality":
                    nominal_metrics["quality_score"],

                "corrected_mean_gap":
                    corrected_metrics["mean_gap"],

                "corrected_max_gap":
                    corrected_metrics["max_gap"],

                "corrected_parallelism":
                    corrected_metrics["parallelism_error"],

                "corrected_quality":
                    corrected_metrics["quality_score"],
            }
        )


df = pd.DataFrame(records)


print("\n==============================================")
print("V3.2 SMALL BATCH SIMULATOR")
print("==============================================")

print(f"\nAssemblies simulated : {N_ASSEMBLIES}")
print(f"Components each      : {N_COMPONENTS}")
print(f"Total assembly steps : {len(df)}")


# ------------------------------------------------------------
# FINAL PRODUCT RESULTS
# ------------------------------------------------------------

final_rows = df[
    df["component_index"] == N_COMPONENTS
].copy()


mean_nominal = final_rows[
    "nominal_quality"
].mean()

mean_corrected = final_rows[
    "corrected_quality"
].mean()


improvement = (
    (mean_nominal - mean_corrected)
    / mean_nominal
    * 100.0
)


print("\nFINAL PRODUCT QUALITY")
print("----------------------------------------------")

print(
    f"Mean nominal final quality   : "
    f"{mean_nominal:.4f}"
)

print(
    f"Mean corrected final quality : "
    f"{mean_corrected:.4f}"
)

print(
    f"Mean improvement             : "
    f"{improvement:.2f}%"
)


if mean_corrected < mean_nominal:
    print(
        "\nPASS - Correction improved mean batch quality."
    )
else:
    print(
        "\nWARNING - Correction did not improve mean quality."
    )


# ------------------------------------------------------------
# SAVE DATA
# ------------------------------------------------------------

output_path = (
    "data/processed/"
    "v3_2_small_batch_component_results.csv"
)

df.to_csv(
    output_path,
    index=False,
)


print(
    f"\nSaved component-level data to:\n"
    f"{output_path}"
)

print("\n==============================================")
print("SMALL BATCH SIMULATION COMPLETED")
print("==============================================")