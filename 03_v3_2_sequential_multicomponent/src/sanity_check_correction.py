import numpy as np

from sequential_assembly import (
    create_initial_state,
    deviation_offset,
    correction_profile,
    update_assembly_state,
    calculate_quality_metrics,
)


print("\n==============================================")
print("V3.2 CORRECTION SANITY CHECK")
print("==============================================\n")


# ------------------------------------------------------------
# CASE 1: OFFSET WITHOUT CORRECTION
# ------------------------------------------------------------

state_no_correction = create_initial_state()

component = deviation_offset(
    offset_mm=0.5
)

state_no_correction = update_assembly_state(
    previous_state=state_no_correction,
    component_deviation=component,
)

metrics_no_correction = calculate_quality_metrics(
    state_no_correction
)

print("CASE 1 - +0.5 mm OFFSET, NO CORRECTION")
print(metrics_no_correction)


# ------------------------------------------------------------
# CASE 2: SAME OFFSET WITH -0.5 mm VERTICAL CORRECTION
# ------------------------------------------------------------

state_with_correction = create_initial_state()

correction = correction_profile(
    z_adj_mm=-0.5,
    theta_adj_deg=0.0,
    locator_offset_mm=0.0,
)

state_with_correction = update_assembly_state(
    previous_state=state_with_correction,
    component_deviation=component,
    correction=correction,
)

metrics_with_correction = calculate_quality_metrics(
    state_with_correction
)

print("\nCASE 2 - +0.5 mm OFFSET, -0.5 mm CORRECTION")
print(metrics_with_correction)


# ------------------------------------------------------------
# SIMPLE AUTOMATIC CHECK
# ------------------------------------------------------------

if metrics_with_correction["quality_score"] < metrics_no_correction["quality_score"]:

    print("\nPASS:")
    print("Correction improved the assembly quality.")

else:

    print("\nFAIL:")
    print("Correction did NOT improve assembly quality.")


print("\n==============================================")
print("CORRECTION SANITY CHECK COMPLETED")
print("==============================================")