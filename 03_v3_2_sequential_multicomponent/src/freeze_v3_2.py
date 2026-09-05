import os
import shutil
import hashlib
import pandas as pd
from datetime import datetime


# ============================================================
# V3.2 FREEZE
# ============================================================
#
# Creates a frozen copy of the current V3.2 implementation,
# validation results, models and notes.
#
# This script does not modify the working project.
# ============================================================


PROJECT_ROOT = os.getcwd()

FREEZE_DIR = os.path.join(
    PROJECT_ROOT,
    "freeze",
    "v3_2_2026_08_26",
)


COPY_ITEMS = [
    "src",
    "models",
    "data/validation",
    "results/validation",
    "results/final_evidence",
    "notes",
]


MANIFEST_FILE = os.path.join(
    FREEZE_DIR,
    "freeze_manifest.csv",
)

README_FILE = os.path.join(
    FREEZE_DIR,
    "README_FREEZE.txt",
)


# ============================================================
# HELPERS
# ============================================================

def file_hash(path):

    hasher = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


# ============================================================
# CREATE FREEZE DIRECTORY
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 FREEZE"
)

print(
    "============================================================"
)


if os.path.exists(
    FREEZE_DIR
):

    raise FileExistsError(
        "\nFreeze directory already exists:\n"
        f"{FREEZE_DIR}\n\n"
        "Nothing was overwritten."
    )


os.makedirs(
    FREEZE_DIR,
    exist_ok=False,
)


# ============================================================
# COPY PROJECT ITEMS
# ============================================================

copied_items = []


for relative_path in COPY_ITEMS:

    source = os.path.join(
        PROJECT_ROOT,
        relative_path,
    )

    destination = os.path.join(
        FREEZE_DIR,
        relative_path,
    )


    if not os.path.exists(
        source
    ):

        print(
            f"WARNING - Missing: {relative_path}"
        )

        continue


    if os.path.isdir(
        source
    ):

        shutil.copytree(
            source,
            destination,
        )

    else:

        os.makedirs(
            os.path.dirname(
                destination
            ),
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )


    copied_items.append(
        relative_path
    )


    print(
        f"Copied: {relative_path}"
    )


# ============================================================
# CREATE FILE MANIFEST
# ============================================================

manifest_records = []


for root, _, files in os.walk(
    FREEZE_DIR
):

    for filename in files:

        full_path = os.path.join(
            root,
            filename,
        )


        if full_path in [
            MANIFEST_FILE,
            README_FILE,
        ]:

            continue


        relative_path = os.path.relpath(
            full_path,
            FREEZE_DIR,
        )


        file_size = os.path.getsize(
            full_path
        )


        sha256 = file_hash(
            full_path
        )


        manifest_records.append(
            {
                "file":
                    relative_path,

                "size_bytes":
                    file_size,

                "sha256":
                    sha256,
            }
        )


manifest_df = pd.DataFrame(
    manifest_records
)


manifest_df = manifest_df.sort_values(
    "file"
)


manifest_df.to_csv(
    MANIFEST_FILE,
    index=False,
)


# ============================================================
# WRITE FREEZE README
# ============================================================

freeze_time = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)


readme_text = f"""
V3.2 TECHNICAL FREEZE

Project:
Machine Learning-Based Optimization of Assembly Parameters
Considering Part Variations

Freeze created:
{freeze_time}

Frozen version:
V3.2 Sequential Multi-Component Adaptive Assembly Framework

Purpose:
This directory preserves the validated technical state used
for final thesis analysis and documentation.

Main frozen controller:
Structured-20 profile-aware Random Forest recommendation.

Bayesian Optimization:
Retained as an optional ambiguity-triggered refinement.

Independent validation:
300 assemblies
1500 sequential component decisions
30 independent batch realizations

Main validation results:
Zero mean final quality       ~ 0.9505
Structured-20 mean quality    ~ 0.2386
Selective BO mean quality     ~ 0.2377

Structured-20 improvement:
~ 74.90%

Structured-20 win vs zero:
~ 99.33%

Robustness:
Minimum multi-seed win        ~ 98.67%
Minimum magnitude win         ~ 99.11%
60% capability win            ~ 98.67%
Minimum quality-weight win    ~ 98.00%

Important modelling interpretation:
The batch disturbance is interpreted as a repeated
batch-dependent assembly-process contribution at each
sequential assembly step.

Validation status:
39/39 result consistency checks passed.

The contents of this folder should not be edited.

Further work should be performed outside this frozen folder.
"""


with open(
    README_FILE,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        readme_text.strip()
        +
        "\n"
    )


# ============================================================
# FINAL CHECK
# ============================================================

required_freeze_items = [
    "src",
    "models",
    "data/validation",
    "results/validation",
    "results/final_evidence",
    "notes",
]


missing_after_copy = []


for relative_path in required_freeze_items:

    path = os.path.join(
        FREEZE_DIR,
        relative_path,
    )

    if not os.path.exists(
        path
    ):

        missing_after_copy.append(
            relative_path
        )


# ============================================================
# RESULT
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "FREEZE SUMMARY"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nFreeze folder:"
)


print(
    FREEZE_DIR
)


print(
    f"\nFiles recorded in manifest : "
    f"{len(manifest_df)}"
)


print(
    f"Copied project sections    : "
    f"{len(copied_items)}"
)


print(
    f"Missing required sections  : "
    f"{len(missing_after_copy)}"
)


if len(
    missing_after_copy
) == 0:

    print(
        "\nRESULT: PASS"
    )

    print(
        "V3.2 technical state has been frozen successfully."
    )

else:

    print(
        "\nRESULT: REVIEW REQUIRED"
    )

    print(
        "Missing frozen sections:"
    )

    for item in missing_after_copy:

        print(
            item
        )


print(
    "\nManifest:"
)


print(
    MANIFEST_FILE
)


print(
    "\nFreeze notes:"
)


print(
    README_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 FREEZE COMPLETED"
)

print(
    "============================================================"
)