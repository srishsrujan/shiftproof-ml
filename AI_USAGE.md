# AI_USAGE.md

## How AI assistance was used

AI assistance was used to accelerate repository scaffolding, documentation drafting, test-case brainstorming and code review.

## Verification policy

Project verification should include:

1. Running the end-to-end training script.
2. Running the automated test suite.
3. Inspecting metric outputs and ensuring the baseline/final models use the same frozen test set.
4. Reviewing calibration, cohort slices, and the generated error report.
5. Running the FastAPI smoke tests.
6. Reviewing the student dashboard, what-if controls, and CSV predictions.

## Human-authored project logic

The following parts are intentionally explicit in the repository:

- `NumpyLogisticRegression` training loop.
- Validation-based model selection.
- Platt calibration calculations.
- Slice-wise evaluation runner.
- Data-generation rules and the synthetic placement target.

AI assistance does not replace understanding of these components; the project owner should be able to explain each one during evaluation.
