# calib_framework

Autonomous superconducting qubit calibration via sequential Bayesian inference and causal reasoning.

Replaces hand-coded FSM condition functions, error code enums, and static retry loops in
[qua-libs](https://github.com/qua-platform/qua-libs) bringup graphs with three principled components:

| Old system | Replacement |
|---|---|
| `PowerRabiErrorCode`, `QubitSpectroscopyErrorCode`, … | `BICDiagnoser` — statistically graded fit quality |
| `bo_optimizer.py` / `bo_node_controller.py` | `GPBayesianOptimizer` + `BONodeController` |
| `should_retry_*` / `should_repeat_*` FSM functions | `BONodeController.should_repeat` + `CausalOrchestrator` |
| `build_resonator_bringup`, `build_qubit_calibration` | `bringup_causal.py` graph |

---

## Table of Contents

- [Installation](#installation)
- [Architecture overview](#architecture-overview)
- [Data flow](#data-flow)
- [Module reference](#module-reference)
  - [core/estimates.py — GaussianEstimate](#coreestimatespy--gaussianestimate)
  - [core/bic.py — BICDiagnoser](#corebicpy--bicdiagnoser)
  - [core/node_result.py — NodeResult](#corenode_resultpy--noderesult)
  - [logging/session_logger.py — SessionLogger](#loggingsession_loggerpy--sessionlogger)
  - [bo/optimizer.py — GPBayesianOptimizer](#booptimizerpy--gpbayesianoptimizer)
  - [bo/node_controller.py — BONodeController](#bonode_controllerpy--bonodecontroller)
  - [causal/discovery.py — CausalGraphLearner](#causaldiscoverypy--causalgraphlearner)
  - [core/orchestrator.py — CausalOrchestrator](#coreorchestratorpy--causalorchestrator)
- [Bringup graph](#bringup-graph)
- [Before and after causal discovery](#before-and-after-causal-discovery)
- [Integration guide](#integration-guide)
- [Configuration reference](#configuration-reference)
- [Development](#development)
- [References](#references)

---

## Installation

```bash
# Minimal install (no qualibrate / causal-learn)
pip install -e .

# With qualibrate integration
pip install -e ".[qualibrate]"

# With causal discovery (requires causal-learn)
pip install -e ".[causal]"

# Everything including dev tools
pip install -e ".[all]"
```

**Using uv:**
```bash
uv pip install -e ".[all]"
```

**Dependencies:**

| Package | Purpose |
|---|---|
| `numpy`, `scipy` | Numerical routines, BIC fitting, EI optimisation |
| `scikit-learn` | `GaussianProcessRegressor` with Matérn 5/2 kernel |
| `networkx` | Causal DAG representation |
| `matplotlib` | DAG visualisation |
| `filelock` | Thread-safe JSONL logging |
| `causal-learn` *(optional)* | GES / PC causal discovery algorithms |
| `qualibrate`, `quam` *(optional)* | QUAlibrate node / graph integration |

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        bringup_causal.py                        │
│                    (QualibrationGraph wrapper)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ uses
          ┌─────────────────▼──────────────────┐
          │         CausalOrchestrator          │
          │  Without DAG: dependency-order retry │
          │  With DAG: causal fault routing     │
          └──────┬────────────────┬────────────┘
                 │                │ reads
     ┌───────────▼──────┐  ┌──────▼──────────────┐
     │  BONodeController │  │  CausalGraphLearner  │
     │  (one per node)   │  │  (offline, ≥30 sess) │
     └───────┬───────────┘  └─────────────────────┘
             │
     ┌───────▼──────────────────────────────────┐
     │              Per iteration                │
     │  1. extract x/y from node.results         │
     │  2. BICDiagnoser.diagnose(x, y)           │
     │  3. BICDiagnoser.to_bo_cost(result)       │
     │  4. GPBayesianOptimizer.register(cost)    │
     │  5. GPBayesianOptimizer.suggest() → QUAM  │
     │  6. GaussianEstimate → QUAM (on success)  │
     │  7. NodeResult → SessionLogger            │
     └───────────────────────────────────────────┘
```

---

## Data flow

```
QUA measurement
      │  raw I/Q xarray Dataset
      ▼
  BICDiagnoser                ← fits null vs signal model, computes ΔBIC
      │  BICResult (ΔBIC, evidence_strength, diagnosis)
      ▼
  to_bo_cost()                ← logistic sigmoid: ΔBIC → [0, 1] BO cost
      │  float cost
      ▼
  GPBayesianOptimizer         ← Matérn 5/2 GP + EI acquisition
  .register(params, cost)     ← updates GP on disk (persistent JSON)
  .suggest()                  ← next parameter suggestion
      │  dict {param: value}
      ▼
  QUAM temp_calibration       ← bo_suggested written here
  [qubit].bo_suggested        ← read by node script on next iteration
      │
      ▼ (on success)
  GaussianEstimate            ← posterior mean + std at best observed point
      │  written to QUAM
      ▼
  QUAM temp_calibration       ← gaussian_estimates[node_id]
  [qubit].gaussian_estimates  ← read by downstream nodes to tighten search
      │
      ▼
  SessionLogger               ← one JSONL line per (node, qubit, iteration)
  bo_state/sessions.jsonl     ← training data for CausalGraphLearner
```

---

## Module reference

### `core/estimates.py` — `GaussianEstimate`

A calibrated parameter value with Gaussian uncertainty. Produced after a successful node run;
stored in `machine.temp_calibration[qubit].gaussian_estimates` as a JSON-serialisable dict.

```python
from calib_framework.core.estimates import GaussianEstimate

est = GaussianEstimate(
    mean=6.123e9,          # posterior mean (physical units)
    std=2.5e6,             # GP posterior std at best observed point
    source_node="02a_resonator_spectroscopy",
    session_id="abc123",
    n_observations=5,
)

print(est.confidence)                  # ∈ [0, 1], 1 = perfectly confident
print(est.is_high_confidence())        # True if confidence ≥ 0.95

# Downstream node: how wide should I sweep?
span = est.search_range(base_range=50e6, k=3.0)
# → base_range + 3σ, covers 99.7% of posterior mass

# Serialise for QUAM storage
d = est.to_dict()          # JSON-compatible dict
est2 = GaussianEstimate.from_dict(d)
```

**Key formula:** `confidence = clamp(1 - std / |mean|, 0, 1)`

**`search_range(base_range, k=3.0)`** returns `base_range + k * std`. Pass the result to
`GPBayesianOptimizer.suggest(upstream_estimates=..., tighten_param_map=...)` to focus the
search around the upstream mean.

---

### `core/bic.py` — `BICDiagnoser`

Replaces all hand-coded error code enums with statistically principled fit diagnosis based on
the Bayesian Information Criterion (Schwarz 1978; Kass & Raftery 1995).

**BIC = −2 · log-likelihood + k · ln(n)**

A lower BIC is better. ΔBIC = BIC(null) − BIC(signal): positive means evidence for a real signal.

| ΔBIC | Evidence strength | Meaning |
|---|---|---|
| > 10 | `strong` | Clearly detected signal |
| 6 – 10 | `moderate` | Signal found (default success threshold) |
| 2 – 6 | `weak` | Marginal signal |
| < 2 | `none` | No signal / noise only |

**Built-in models:**

| Node type | Null model | Signal model |
|---|---|---|
| `resonator_spectroscopy` | Constant | Lorentzian (k=4) |
| `resonator_punch_out` | Constant | Lorentzian (k=4) |
| `qubit_spectroscopy_vs_power` | Constant | Lorentzian (k=4) |
| `power_rabi`, `time_rabi` | Constant | Damped cosine (k=5) |
| `t1` | Constant | Exponential decay (k=3) |

```python
import numpy as np
from calib_framework.core.bic import BICDiagnoser

diagnoser = BICDiagnoser(node_type="power_rabi")

# x = amplitude sweep, y = measured Rabi population
result = diagnoser.diagnose(x=np.linspace(0, 0.5, 50), y=measured_data)

print(result.winning_model)       # "DampedCosineModel"
print(result.delta_bic)           # e.g. 14.3
print(result.evidence_strength)   # "strong"
print(result.diagnosis)           # "strong_signal_found"

# Convert to BO cost (0 = perfect, 1 = failure)
cost = diagnoser.to_bo_cost(result)   # e.g. 0.04
```

**Adding a custom model:**
```python
from calib_framework.core.bic import BICDiagnoser, ConstantModel

class MyModel:
    name = "MyModel"
    n_params = 3
    def fit(self, x, y): ...
    def log_likelihood(self, x, y, params): ...

BICDiagnoser.MODEL_REGISTRY["my_node_type"] = [ConstantModel(), MyModel()]
```

---

### `core/node_result.py` — `NodeResult`

Typed record of one node execution. Produced by `BONodeController.should_repeat()` and
written as one JSONL line by `SessionLogger`. Contains everything needed for post-hoc analysis
and causal discovery.

```python
from calib_framework.core.node_result import NodeResult

# Serialise / deserialise
d = node_result.to_dict()
node_result2 = NodeResult.from_dict(d)

# Derive outcome from BICResult
outcome = NodeResult.outcome_from_bic(bic_result)
# "successful" if evidence_strength in {"strong", "moderate"}
# "uncertain"  if evidence_strength == "weak"
# "failed"     if evidence_strength == "none"
```

**Fields:** `node_id`, `qubit`, `session_id`, `timestamp`, `parameters_used`, `raw_fit`,
`bic_result`, `bo_cost`, `upstream_estimates`, `output_estimate`, `outcome`, `retry_count`,
`n_shots`, `wall_clock_seconds`.

---

### `logging/session_logger.py` — `SessionLogger`

Appends structured JSONL records to `{bo_state_dir}/sessions.jsonl`. Thread-safe via `filelock`.
This file is the training data for `CausalGraphLearner` and the empirical validation dataset.

```python
from calib_framework.logging.session_logger import SessionLogger

log = SessionLogger(
    bo_state_dir="bo_state/",
    session_id="abc123",
    cooldown_id="DR3-Run009",
    fridge_id="DR3",
)

log.log_session_start(qubits=["q1", "q2"], graph_name="bringup_causal")
log.log_node_result(node_result)           # called by BONodeController
log.log_session_end({"q1": "successful", "q2": "failed"})

# Load for analysis
records = SessionLogger.load_sessions("bo_state/sessions.jsonl")

# Build outcome matrix for causal discovery
matrix = SessionLogger.to_outcome_matrix(
    records,
    node_ids=["02a_resonator_spectroscopy", "03c_qubit_spectroscopy_vs_power", "04b_power_rabi"],
    metric="bo_cost",   # or "outcome"
)
# shape: (n_sessions, n_nodes)
```

**JSONL record format** (one JSON object per line):
```json
{
  "_record_type": "node_result",
  "_timestamp": "2026-05-28T00:00:00+00:00",
  "session_id": "abc123",
  "cooldown_id": "DR3-Run009",
  "fridge_id": "DR3",
  "node_id": "04b_power_rabi",
  "qubit": "q1",
  "outcome": "successful",
  "bo_cost": 0.04,
  "bic_result": {"winning_model": "DampedCosineModel", "delta_bic": 14.3, ...},
  "output_estimate": {"mean": 0.25, "std": 0.01, "confidence": 0.96, ...},
  ...
}
```

---

### `bo/optimizer.py` — `GPBayesianOptimizer`

GP-BO with Matérn 5/2 kernel (scikit-learn) and Expected Improvement acquisition
(scipy `differential_evolution`). Warm-starts from disk; observation files are
**backward-compatible** with the old `bo_state/*.json` schema.

```python
from calib_framework.bo.optimizer import GPBayesianOptimizer, ParameterBound

optimizer = GPBayesianOptimizer(
    node_key="04b_power_rabi",
    qubit="q1",
    bounds=[
        ParameterBound("amplitude", low=0.05, high=0.95),
    ],
    bo_state_dir="bo_state/",
    n_initial_random=3,    # LHS samples before GP fits
)

# Suggest next parameters (Latin Hypercube until n_initial_random, then EI)
suggestion = optimizer.suggest()   # {"amplitude": 0.312}

# Register an observation (cost from BICDiagnoser.to_bo_cost())
optimizer.register(params={"amplitude": 0.312}, cost=0.23)

print(optimizer.best_params)          # {"amplitude": 0.25}
print(optimizer.best_cost)            # 0.04
print(optimizer.posterior_std_at_best)  # GP std at best point → GaussianEstimate.std
```

**Tightening bounds with upstream estimates:**
```python
suggestion = optimizer.suggest(
    upstream_estimates={"02a_resonator_spectroscopy": est},
    tighten_param_map={"center_freq": "02a_resonator_spectroscopy"},
)
# The "center_freq" bound is shrunk to [mean - search_range/2, mean + search_range/2]
```

**ParameterBound:**
```python
ParameterBound("log_power", low=1e-4, high=1.0, log_scale=True)
# Optimisation is performed in log-space; physical values are always returned
```

---

### `bo/node_controller.py` — `BONodeController`

The main integration point with QUAlibrate. Its `should_repeat(node, target)` method is the
drop-in `on=` argument for `graph.loop()`.

```python
from calib_framework.bo.node_controller import BONodeController
from calib_framework.bo.optimizer import ParameterBound
from calib_framework.logging.session_logger import SessionLogger

session_id = str(uuid.uuid4())
log = SessionLogger(bo_state_dir="bo_state/", session_id=session_id,
                    cooldown_id="DR3-Run009", fridge_id="DR3")

controller = BONodeController(
    node_key="04b_power_rabi",
    node_type="power_rabi",              # key in BICDiagnoser.MODEL_REGISTRY
    bounds=[ParameterBound("amplitude", 0.05, 0.95)],
    machine=machine,                     # QUAM Quam instance
    bo_state_dir="bo_state/",
    session_id=session_id,
    logger_inst=log,
    max_iterations=8,
    success_delta_bic=6.0,               # "moderate" evidence threshold
    x_axis_key="amplitude",              # key in node.results["ds_raw"][qubit]
    y_axis_key="state",
)

# Use as QUAlibrate loop condition:
graph.loop(
    power_rabi_node,
    on=controller.should_repeat,
    max_iterations=8,
)
```

**What `should_repeat` does on each call:**
1. Extracts x/y from `node.results["ds_raw"][target]`
2. Runs `BICDiagnoser.diagnose(x, y)` → `BICResult`
3. Converts to BO cost → registers with `GPBayesianOptimizer`
4. Suggests next params → writes to `machine.temp_calibration[target].bo_suggested`
5. On success: writes `GaussianEstimate` to `machine.temp_calibration[target].gaussian_estimates`
6. Logs `NodeResult` to `SessionLogger`
7. Returns `True` (retry) or `False` (done)

**QUAM fields written:**

| Field | When | Read by |
|---|---|---|
| `temp_calibration[q].bo_suggested[node_key]` | Every retry | Node script's `create_qua_program` |
| `temp_calibration[q].gaussian_estimates[node_id]` | On success | Downstream `BONodeController.suggest()` |

---

### `causal/discovery.py` — `CausalGraphLearner`

Learns a causal DAG from calibration session history using GES (Greedy Equivalence Search)
or PC (Peter-Clark) algorithms from the `causal-learn` library.

```python
from calib_framework.causal.discovery import CausalGraphLearner
from calib_framework.logging.session_logger import SessionLogger

node_sequence = [
    "02a_resonator_spectroscopy",
    "02b_resonator_punch_out",
    "03c_qubit_spectroscopy_vs_power",
    "04b_power_rabi",
]

learner = CausalGraphLearner(
    node_sequence=node_sequence,
    algorithm="GES",    # or "PC"
    alpha=0.05,         # significance level (PC only)
)

records = SessionLogger.load_sessions("bo_state/sessions.jsonl")
matrix = SessionLogger.to_outcome_matrix(records, node_ids=node_sequence)

dag = learner.fit(matrix)   # nx.DiGraph; requires ≥ 15 sessions (warns below 30)

# Validate against physical priors
warnings = learner.validate_physical_consistency()
for w in warnings:
    print(w)

# Save / load
learner.save("causal_dag.json")
dag = CausalGraphLearner.load("causal_dag.json")

# Plot
learner.plot("causal_dag.png")
```

**Background knowledge:** temporal ordering is automatically enforced — edges from later nodes
to earlier nodes are forbidden, halving the search space.

---

### `core/orchestrator.py` — `CausalOrchestrator`

DAG-aware calibration orchestrator. Two operating modes:

**Dependency-order mode** (`causal_dag=None`): on failure, re-run the immediate predecessor in `node_sequence`.

**Causal routing mode** (`causal_dag` provided): queries the learned DAG + upstream `GaussianEstimate`
uncertainties to identify the most likely root-cause node.

```python
from calib_framework.core.orchestrator import CausalOrchestrator
import networkx as nx

dag = CausalGraphLearner.load("causal_dag.json")

orchestrator = CausalOrchestrator(
    node_sequence=BRINGUP_NODE_SEQUENCE,
    bo_controllers={
        "02a_resonator_spectroscopy": controller_02a,
        "03c_qubit_spectroscopy_vs_power": controller_03c,
        "04b_power_rabi": controller_04b,
    },
    causal_dag=dag,    # None for dependency-order mode
    machine=machine,
    max_upstream_retries=2,
)
```

---

## Bringup graph

`qualibrate_graphs/bringup_causal.py` is the top-level QUAlibrate graph.
It replaces `92_calibration_graph_bringup_fixed_frequency_transmon_adaptive.py`
(frozen as `92_BASELINE_bringup_fsm_frozen.py`).

**Nominal execution sequence:**
```
02a_resonator_spectroscopy
    → 02b_resonator_punch_out
        → 03c_qubit_spectroscopy_vs_power
            → 04b_power_rabi
```

**Parameters (`BringUpCausalParameters`):**

| Parameter | Default | Description |
|---|---|---|
| `qubits` | `[]` | Qubit names to calibrate |
| `bo_state_dir` | `"bo_state"` | GP observation + session log directory |
| `causal_dag_path` | `None` | Path to DAG JSON; enables causal routing when set |
| `max_bo_iterations` | `8` | Max BO retries per node per qubit |
| `success_delta_bic` | `6.0` | ΔBIC threshold for success ("moderate") |
| `resonator_frequency_span_mhz` | `100.0` | Resonator spectroscopy sweep span |
| `resonator_n_shots` | `1000` | Resonator measurement shots |
| `punch_out_power_span_dbm` | `30.0` | Punch-out power sweep span |
| `qubit_frequency_span_mhz` | `200.0` | Qubit spectroscopy sweep span |
| `rabi_amplitude` | `0.5` | Rabi base π-pulse amplitude |

---

## Before and after causal discovery

### Before a causal DAG exists (dependency-order mode)

Run the bringup graph with `causal_dag_path=None`. The system uses BO + BIC with
dependency-order retry: on failure, the immediate predecessor is re-run.
Session data accumulates in `bo_state/sessions.jsonl`.

```bash
# In QUAlibrate GUI: select "bringup_causal", leave causal_dag_path empty
```

### Learning the causal DAG (≥ 30 sessions)

```bash
python scripts/analyze_causal_dag.py \
    --sessions bo_state/sessions.jsonl \
    --output causal_dag.json \
    --algorithm GES \
    --plot causal_dag.png
```

This prints a summary of found edges and any physical consistency warnings.

### After causal discovery (causal routing mode)

Set `causal_dag_path = "causal_dag.json"` in the graph parameters. The orchestrator now
uses the learned DAG to route failures to the most likely root-cause node instead of
always retrying the immediate predecessor.

---

## Integration guide

### Connecting a node script to BO suggestions

In your node's `create_qua_program`, read the BO suggestion from QUAM if present:

```python
def create_qua_program(node):
    machine = node.machine
    for q in node.namespace["qubits"]:
        # Read BO suggestion (written by BONodeController.should_repeat)
        temp = machine.temp_calibration.get(q.name)
        suggested = getattr(temp, "bo_suggested", {}) or {}
        node_suggestion = suggested.get("04b_power_rabi", {})

        amplitude = node_suggestion.get("amplitude", node.parameters.max_amp)
        # ... use amplitude in the QUA program
```

### Consuming upstream GaussianEstimates

In a downstream node's `create_qua_program`, tighten the sweep window:

```python
from calib_framework.core.estimates import GaussianEstimate

temp = machine.temp_calibration.get(q.name)
ge_dict = getattr(temp, "gaussian_estimates", {}) or {}
upstream = ge_dict.get("02a_resonator_spectroscopy")
if upstream:
    est = GaussianEstimate.from_dict(upstream)
    span_hz = est.search_range(base_range=100e6)   # [mean ± span/2]
    center_hz = est.mean
```

### QUAM prerequisites

`TemporaryCalibrationData` in `quam_config/my_quam.py` must have the `gaussian_estimates` field:

```python
@quam_dataclass
class TemporaryCalibrationData(QuamComponent):
    # ... existing fields ...
    gaussian_estimates: Optional[Dict[str, Any]] = None
    bo_suggested: Optional[Dict[str, Any]] = None
```

Both fields are already added as part of the qua-libs migration in `my_quam.py`.

---

## Configuration reference

### BICDiagnoser node types

```python
BICDiagnoser.MODEL_REGISTRY.keys()
# "resonator_spectroscopy", "resonator_punch_out", "chi_shift",
# "qubit_spectroscopy_vs_power", "power_rabi", "time_rabi", "t1"
```

### ParameterBound

```python
ParameterBound(name: str, low: float, high: float, log_scale: bool = False)
```

Set `log_scale=True` for amplitude parameters that span orders of magnitude.
Requires `low > 0`.

### BONodeController axis keys

Common `x_axis_key` / `y_axis_key` values (must match coordinate/variable names in
`node.results["ds_raw"][qubit]`):

| Node | x_axis_key | y_axis_key |
|---|---|---|
| `02a_resonator_spectroscopy` | `"detuning"` or `"frequency"` | `"I"` or `"amplitude"` |
| `02e_resonator_punch_out` | `"frequency"` | `"amplitude"` |
| `03c_qubit_spectroscopy_vs_power` | `"detuning"` | `"state"` |
| `04b_power_rabi` | `"amplitude"` | `"state"` |

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=calib_framework --cov-report=term-missing

# Lint
ruff check calib_framework/
```

### Running tests without QUAlibrate / QUAM

All tests in `tests/` are standalone and do not require QUAlibrate or QUAM:

```bash
pytest tests/test_bic.py -v
```

---

## References

- **BIC model selection:** Schwarz, G. (1978). "Estimating the Dimension of a Model." *Annals of Statistics*, 6(2), 461–464.
- **BIC evidence thresholds:** Kass, R.E. & Raftery, A.E. (1995). "Bayes Factors." *JASA*, 90(430), 773–795.
- **GP-BO / EI acquisition:** Snoek, J., Larochelle, H., & Adams, R.P. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms." *NeurIPS*. [arXiv:1206.2944](https://arxiv.org/abs/1206.2944)
- **GP textbook:** Rasmussen, C.E. & Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning.* MIT Press.
- **Matérn kernel:** Matérn, B. (1960). *Spatial Variation.* Springer.
- **Causal DAG calibration:** Kelly, J. et al. (2018). "Physical qubit calibration on a directed acyclic graph." [arXiv:1803.03226](https://arxiv.org/abs/1803.03226)
- **Causal BO:** Aglietti, V. et al. (2020). "Causal Bayesian Optimization." *AISTATS*. [arXiv:2006.01085](https://arxiv.org/abs/2006.01085)
- **GES algorithm:** Chickering, D.M. (2002). "Optimal Structure Identification With Greedy Search." *JMLR*, 3, 507–554.
- **PC algorithm:** Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search.* MIT Press.
- **causal-learn library:** [https://github.com/py-why/causal-learn](https://github.com/py-why/causal-learn)
