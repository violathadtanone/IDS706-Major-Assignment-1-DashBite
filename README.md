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

Install dependencies and run the automated test suite:

```bash
make install
make test
```

Start the full application as persistent background processes:

```bash
make run
```

The default polling interval is 15 seconds. Stage logs are written to `.logs/` and PID records to `.logs/pids/`. New batches flow through `data/raw/`, `data/features/`, and `data/predictions/`. Open Model Pulse at <http://localhost:8501>.

Stop the full stack with:

```bash
make stop
```

## Smoke Test Results

## Smoke Test Results

The project was exercised using the documented setup and Docker Compose workflow.

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

## Evaluation & Reflection
