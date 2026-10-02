# IDS 706 1st Major Assignment: AI-Assisted Development Workflow - 1 Oct 2026

## Project Description
This is 1st major assignment under IDS 706 with the purpose to practice the development workflow with AI-assitance tool. The selected project for this assignment is 'Extend and Containerize DashBite'.

DashBite is a file-based application that demonstrates an end-to-end late-order prediction pipeline. It generates synthetic food orders, validates and transforms them into model features, trains a logistic-regression classifier, and writes late-risk predictions. Model Pulse presents recent order volume, predicted late rates, and input-quality failures in a live dashboard. Each stage runs independently and exchanges artifacts through the `data/` directories.

## Project Structure

```text
DashBite/
├── requirements.txt                   # Python dependencies
├── Makefile                            # Install, test, stage, and full-stack commands
├── Dockerfile                          # Container image definition
├── compose.yaml                        # Multi-service container configuration
├── pytest.ini                          # Pytest settings and test markers
├── README.md                           # Project overview and usage
├── docs/
│   └── plan.md                         # Project stages, architecture, and smoke tests
├── pipeline/
│   ├── config.py                       # Environment-backed application settings
│   ├── paths.py                        # Shared data-directory resolution
│   ├── simulator.py                    # Synthetic order generation
│   ├── preprocess.py                   # Input validation and feature preparation
│   ├── train.py                        # Model training and checkpoint publication
│   ├── infer.py                        # Prediction generation from published models
│   ├── pulse.py                        # Dashboard analytics helpers
│   ├── dashboard.py                    # Streamlit Model Pulse application
│   └── process_manager.py              # Persistent local stack start and stop
├── data/
│   ├── raw/                            # Simulated order batches
│   ├── features/                       # Validated feature batches
│   ├── models/                         # Versioned model checkpoints and metrics
│   ├── predictions/                    # Per-batch prediction files
│   └── quality/                        # Per-batch validation summaries
└── tests/
    ├── unit/                           # Focused component tests
    ├── regression/                     # Contract and behavior regression tests
    └── integration/                    # Pipeline and application integration tests
```

## Run the Application
The Makefile provides the main classroom-facing commands for running and testing the pipeline.
- Install dependencies and run the automated test suite:

```bash
make install
make test
```

- Run the full application as persistent background processes:

```bash
make run
```

- Individual stages can also be run with:

```bash
make simulator
make preprocess
make train
make infer
make dashboard
```

The default polling interval is 15 seconds. Stage logs are written to `.logs/` and PID records to `.logs/pids/`. New batches flow through `data/raw/`, `data/features/`, and `data/predictions/`. Open Model Pulse at <http://localhost:8501>.

Stop the full stack with:

```bash
make stop
```

## Smoke Test Results
Smoke Test was executed with the following results:

**Passed**
- **Setup:** `make install` installed all requirements successfully.
- **Tests:** `make test` passed (109 tests).
- **Image build:** `docker compose build` completed successfully.
- **Pipeline:** `docker compose up -d` started the simulator, preprocessing, training, and inference stages. Raw, feature, model, prediction, and quality artifacts were all present in the shared data volume.
- **Model Pulse UI:** The page rendered at `http://localhost:8501` with its KPIs and charts when reachable.

**Failed**
- **Compose dashboard service:** The container exited with status 2 because its configured target, `pipeline/dashboard/app.py`, does not exist in the image.

**Conclusion**
- Setup, tests, image build, and artifact production succeeded, but the full container smoke test did not pass because the dashboard service could not stay running.
- Next step: correct the dashboard entrypoint and rerun the Compose smoke test before treating the containerized app as verified.

**Note:** Individual targets (`make simulator`, `make preprocess`, `make train`, `make infer`, `make dashboard`) run their stage in the foreground.

## Docker

## Evaluation & Reflection
The selected project for this assignment is 'Extend and Containerize DashBite'.

### 1. AI Assitants
AI Assitants play 3 key major roles for the project developement. This includes:
- **Architecture** - This AI supported in defining the project requirements, architecture, implementation plan, risks, and verification approach.
- **Builder:** - This AI supported the implementation of the plan, writes or updates the code and tests, and makes corrections based on review.
**Checker/Tester:** - This AI acted as an independent auditor to check the implementation against the plan, tests edge cases and system behavior, and fixes or reports any issues found.

### 2. Accepted AI recomendations
Key examples of recommendations accepted include:
- **Unique order IDs** - Accepted the Tester’s recommendation to make order IDs unique across simulator batches while keeping the seeded output reproducible.
- **Separate quality sidecar** - Accepted the Architect’s recommendation to store data-quality failure counts separately, so the existing feature CSV format would not need to change.

### 3. Adjustments to AI recommendations
Key examples of adjustments made to recommendations from AI include:
- **"No defect" verdict on the dashboard** - Rejected the Tester’s conclusion after checking the dashboard and finding that the KPI numbers were not visible. After requesting for visual inspection, the tester later identified the near-white text on white cards and led to a CSS fix.
- **initial 'was_late' validation** - Requested the Builder to tighten the preprocessing rules so that was_late accepts only true, false, 1, and 0, rather than allowing other values such as yes or no.

### Final Result Inspection
Examples for independent verification of the final results include:
- **Inference independence check** - Verified that inference could continue using an existing checkpoint after training stopped, and that a newly published checkpoint was used only for later feature batches.
- **Visual dashboard check:** - Opened Model Pulse and identified that the KPI values were not visible even though valid orders existed then proceeded to tell AI for issue resolution.


