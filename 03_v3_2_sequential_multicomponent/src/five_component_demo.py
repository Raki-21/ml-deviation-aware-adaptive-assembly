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


def print_metrics(step_name, metrics):
    print(
        f"{step_name:<22} "
        f"Mean Gap: {metrics['mean_gap']:.4f} | "
        f"Max Gap: {metrics['max_gap']:.4f} | "
        f"Parallelism: {metrics['parallelism_error']:.4f} | "
        f"Quality: {metrics['quality_score']:.4f}"
    )


# ============================================================
# DEFINE THE SAME FIVE COMPONENTS FOR BOTH STRATEGIES
# ============================================================

components = [
    deviation_offset(offset_mm=0.50),
    deviation_tilt(angle_deg=0.02),
    deviation_bend(amplitude_mm=0.30),
    deviation_waviness(amplitude_mm=0.15, waves=3),
    deviation_twist(amplitude_mm=0.20),
]


# ============================================================
# RUN A - NO CORRECTION
# ============================================================

print("\n============================================================")
print("RUN A - FIVE COMPONENT ASSEMBLY WITHOUT CORRECTION")
print("============================================================\n")

state_no_correction = create_initial_state()

no_correction_results = []

for i, component in enumerate(components, start=1):

    state_no_correction = update_assembly_state(
        previous_state=state_no_correction,
        component_deviation=component,
    )

    metrics = calculate_quality_metrics(
        state_no_correction
    )

    no_correction_results.append(metrics)

    print_metrics(
        f"After Component {i}",
        metrics,
    )


# ============================================================
# RUN B - SIMPLE MANUAL CORRECTION
# ============================================================

print("\n============================================================")
print("RUN B - FIVE COMPONENT ASSEMBLY WITH MANUAL CORRECTION")
print("============================================================\n")

state_corrected = create_initial_state()

corrected_results = []


# Component 1: +0.50 mm offset
correction_1 = correction_profile(
    z_adj_mm=-0.50,
)

state_corrected = update_assembly_state(
    previous_state=state_corrected,
    component_deviation=components[0],
    correction=correction_1,
)

metrics = calculate_quality_metrics(state_corrected)
corrected_results.append(metrics)
print_metrics("After Component 1", metrics)


# Component 2: +0.02 degree tilt
correction_2 = correction_profile(
    theta_adj_deg=-0.02,
)

state_corrected = update_assembly_state(
    previous_state=state_corrected,
    component_deviation=components[1],
    correction=correction_2,
)

metrics = calculate_quality_metrics(state_corrected)
corrected_results.append(metrics)
print_metrics("After Component 2", metrics)


# Component 3: bend
# No exact rigid-body correction exists for bend.
# Apply a small vertical correction only.
correction_3 = correction_profile(
    z_adj_mm=-0.15,
)

state_corrected = update_assembly_state(
    previous_state=state_corrected,
    component_deviation=components[2],
    correction=correction_3,
)

metrics = calculate_quality_metrics(state_corrected)
corrected_results.append(metrics)
print_metrics("After Component 3", metrics)


# Component 4: waviness
# Global correction cannot fully remove local waviness.
correction_4 = correction_profile(
    locator_offset_mm=-0.08,
)

state_corrected = update_assembly_state(
    previous_state=state_corrected,
    component_deviation=components[3],
    correction=correction_4,
)

metrics = calculate_quality_metrics(state_corrected)
corrected_results.append(metrics)
print_metrics("After Component 4", metrics)


# Component 5: twist
# Existing three-parameter correction has limited ability
# to fully compensate twist.
correction_5 = correction_profile(
    theta_adj_deg=-0.005,
)

state_corrected = update_assembly_state(
    previous_state=state_corrected,
    component_deviation=components[4],
    correction=correction_5,
)

metrics = calculate_quality_metrics(state_corrected)
corrected_results.append(metrics)
print_metrics("After Component 5", metrics)


# ============================================================
# FINAL COMPARISON
# ============================================================

final_no_correction = no_correction_results[-1]["quality_score"]
final_corrected = corrected_results[-1]["quality_score"]

improvement = (
    (final_no_correction - final_corrected)
    / final_no_correction
    * 100.0
)

print("\n============================================================")
print("FINAL COMPARISON")
print("============================================================")

print(
    f"No-correction final quality score : "
    f"{final_no_correction:.4f}"
)

print(
    f"Corrected final quality score     : "
    f"{final_corrected:.4f}"
)

print(
    f"Quality improvement               : "
    f"{improvement:.2f}%"
)

if final_corrected < final_no_correction:
    print("\nPASS - Sequential correction reduced accumulated error.")
else:
    print("\nFAIL - Correction strategy requires inspection.")

print("\n============================================================")
print("FIVE COMPONENT DEMO COMPLETED")
print("============================================================")