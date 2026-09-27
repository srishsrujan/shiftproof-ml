# Failure log

The training pipeline creates `reports/failure_analysis.csv` with at least 20 analysed mistakes when the test set contains at least 20 errors. Review the rows and add human comments before final submission.

Suggested columns for manual review:

- request_id
- actual late/on-time label
- probability
- confidence
- request type
- priority
- queue length
- system load
- estimated work hours
- likely cause
- human interpretation
- proposed fix
