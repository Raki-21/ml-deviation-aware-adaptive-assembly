from sequential_assembly import (
    create_initial_state,
    deviation_local_bump,
    correction_profile,
    update_assembly_state,
    calculate_quality_metrics,
)


print("\n==============================================")
print("V3.2 LOCATOR CORRECTION SANITY CHECK")
print("==============================================\n")


# ------------------------------------------------------------
# CASE 1 - LOCAL DEVIATION WITHOUT CORRECTION
# ------------------------------------------------------------

state = create_initial_state()

component = deviation_local_bump(
    amplitude_mm=0.4,
    sigma=0.12,
)

state_no_correction = update_assembly_state(
    previous_state=state,
    component_deviation=component,
)

metrics_no_correction = calculate_quality_metrics(
    state_no_correction
)

print("CASE 1 - LOCAL +0.4 MM DEVIATION, NO CORRECTION")
print(metrics_no_correction)


# ------------------------------------------------------------
# CASE 2 - SAME DEVIATION WITH OPPOSITE LOCATOR CORRECTION
# ------------------------------------------------------------

state = create_initial_state()

correction = correction_profile(
    z_adj_mm=0.0,
    theta_adj_deg=0.0,
    locator_offset_mm=-0.4,
    locator_sigma=0.12,
)

state_with_correction = update_assembly_state(
    previous_state=state,
    component_deviation=component,
    correction=correction,
)

metrics_with_correction = calculate_quality_metrics(
    state_with_correction
)

print("\nCASE 2 - LOCAL +0.4 MM DEVIATION, -0.4 MM LOCATOR CORRECTION")
print(metrics_with_correction)


# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------

if metrics_with_correction["max_gap"] < metrics_no_correction["max_gap"]:
    print("\nPASS - Locator correction reduced the local mismatch.")
else:
    print("\nFAIL - Locator correction logic needs inspection.")


print("\n==============================================")
print("LOCATOR SANITY CHECK COMPLETED")
print("==============================================")