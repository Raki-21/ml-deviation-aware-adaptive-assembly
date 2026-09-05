from sequential_assembly import (
    create_initial_state,
    deviation_tilt,
    correction_profile,
    update_assembly_state,
    calculate_quality_metrics,
)


print("\n==============================================")
print("V3.2 TILT CORRECTION SANITY CHECK")
print("==============================================\n")


# CASE 1 - TILT WITHOUT CORRECTION

state = create_initial_state()

component = deviation_tilt(
    angle_deg=0.02
)

state_no_correction = update_assembly_state(
    previous_state=state,
    component_deviation=component,
)

metrics_no_correction = calculate_quality_metrics(
    state_no_correction
)

print("CASE 1 - +0.02 DEG TILT, NO CORRECTION")
print(metrics_no_correction)


# CASE 2 - SAME TILT WITH OPPOSITE ANGULAR CORRECTION

state = create_initial_state()

correction = correction_profile(
    z_adj_mm=0.0,
    theta_adj_deg=-0.02,
    locator_offset_mm=0.0,
)

state_with_correction = update_assembly_state(
    previous_state=state,
    component_deviation=component,
    correction=correction,
)

metrics_with_correction = calculate_quality_metrics(
    state_with_correction
)

print("\nCASE 2 - +0.02 DEG TILT, -0.02 DEG CORRECTION")
print(metrics_with_correction)


# RESULT

if metrics_with_correction["parallelism_error"] < metrics_no_correction["parallelism_error"]:
    print("\nPASS - Angular correction reduced parallelism error.")
else:
    print("\nFAIL - Angular correction logic needs inspection.")


print("\n==============================================")
print("TILT SANITY CHECK COMPLETED")
print("==============================================")