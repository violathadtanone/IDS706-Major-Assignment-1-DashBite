# IDS706-Mini-Assignment-4-DashBite

## Running the demo

Install dependencies and run the automated suite with `make install` and `make test`. Each pipeline stage is a separate process started in its own terminal: `make simulator`, `make preprocess`, `make train`, `make infer`, and `make dashboard`. They exchange files under `data/`; after training publishes a checkpoint, inference can run with training stopped. Stop live stages with Ctrl+C, in reverse order.