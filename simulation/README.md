# ProjectCanary-Sim

Synthetic scenario simulator (schema, scenario engine, one end-to-end run). Independent of Canary-Analyst.

```
pip install -r requirements.txt
python -m pytest -q
python scripts/run_one.py [--seed N]
```
Docs: `docs/schema_v1.md` (frozen), `docs/data_dictionary.md`.

Telemetry / faults / confounders (see docs/report_sensors_faults.md):
```n python scripts/run_faults_demo.py
```
