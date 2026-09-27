# AI_USAGE.md

## How AI assistance was used

AI assistance was used to accelerate repository scaffolding, documentation drafting, test-case brainstorming and code review.

## Verification policy

Every generated component was checked by:

1. Running the end-to-end training script.
2. Running the automated test suite.
3. Inspecting metric outputs and ensuring the baseline/final models use the same frozen test set.
4. Manually checking the calibration and abstention calculations.
5. Manually checking the drift report on shifted synthetic data.
6. Running the FastAPI smoke tests.
7. Reviewing the generated dashboard and API schemas.

## Human-authored project logic

The following parts are intentionally explicit in the repository:

- `NumpyLogisticRegression` training loop.
- Validation-based confidence/abstention search.
- Platt calibration calculations.
- Slice-wise evaluation runner.
- PSI drift calculations.
- Robustness-test orchestration.
- Data-generation rules and target logic.

AI assistance does not replace understanding of these components; the project owner should be able to explain each one during evaluation.
