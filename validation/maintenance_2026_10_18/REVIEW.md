# Additional local CPU execution

The unchanged current implementation at `d49c74260ee0a6d51f794c6e9210030d4d086fea`
passed all 17 numerical harness experiments in this Python 3.12 environment.
The available unit suite passed 100 tests with three optional dependency skips.
`cpu_results.json` records numerical library versions and all experiment values;
`source_sha256.json` fingerprints the implementation and harness used.

This is another local CPU execution, not an observed hosted Colab rerun.
The old failed direct-rejection coverage experiment remains failed; no empirical
cardiac uncertainty calibration, fresh patient cohort or clinical qualification
is established. No inference algorithm, threshold or seed was changed.
