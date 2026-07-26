# Overview

## Description

The **ExpansionRx–OpenADMET Blind Challenge** asks you to predict **ADMET** (Absorption,
Distribution, Metabolism, Excretion, Toxicity) properties of small molecules from their
chemical structure alone.

The data is real-world drug-discovery data contributed by **Expansion Therapeutics** from
campaigns on RNA-mediated diseases — not a public benchmark. Each molecule is given as a
SMILES string, and the task is to predict **9 experimentally measured ADMET endpoints**.

Accurate ADMET prediction is one of the highest-value problems in early drug discovery:
compounds that are potent but poorly absorbed, rapidly cleared, or heavily protein-bound
fail late and expensively. The endpoints here span solubility, lipophilicity, metabolic
stability, permeability/efflux, and tissue binding.

## Evaluation

Predictions are scored **per endpoint** with **Mean Absolute Error (MAE)**, and the overall
ranking uses **MA-RAE (Macro-Averaged Relative Absolute Error)**. **Lower is better.**

The official scoring procedure, per endpoint:

1. **Transform.** Every endpoint **except `LogD`** is clipped at 0 and log-transformed:

   ```python
   y = np.clip(y, a_min=0, a_max=None)
   y = np.log10(y + 1)
   ```

   `LogD` is already on a log scale and is used **as-is**.

2. **Metrics** on the transformed values:

   ```
   MAE = mean(|y_pred - y_true|)
   RAE = MAE / mean(|y_true - mean(y_true)|)
   ```

   (`R2`, `Spearman R` and `Kendall's Tau` are also reported, but do not determine the
   ranking.) Metrics are bootstrapped over 1000 resamples and averaged.

3. **Macro average.** `MA-RAE` is the unweighted mean of `RAE` over all 9 endpoints — every
   endpoint counts equally regardless of how many molecules have a measurement for it.

Because `RAE` normalises by each endpoint's own spread, it is comparable across endpoints
with wildly different units and ranges.

### Submission File

Submit a CSV with the `Molecule Name` column plus one column per endpoint, using the exact
column names below. **Predictions must be in the original measurement units** — the scorer
applies the log transform itself, so do **not** submit log-transformed values.

Every molecule in the test set must appear exactly once (no missing rows, no duplicates),
and every endpoint column must contain valid numeric predictions.

```
Molecule Name,LogD,KSOL,MLM CLint,HLM CLint,Caco-2 Permeability Efflux,Caco-2 Permeability Papp A>B,MPPB,MBPB,MGMB
E-0012345,1.8,120.0,250.0,20.0,1.5,12.0,5.0,2.0,3.0
...
```

# Dataset Description

Molecules are identified by `Molecule Name` and described by a single `SMILES` string. All
predictive information must be derived from the SMILES (e.g. RDKit descriptors, fingerprints,
graph representations, or pretrained molecular models). No other input features are provided.

The test set is **blinded**: it contains `Molecule Name` and `SMILES` but no measured values.
You must therefore build your own validation split from the training data to estimate
performance.

## File descriptions

- **train.csv** — training set: `Molecule Name`, `SMILES`, and the 9 endpoint columns.
- **test.csv** — blinded test set: `Molecule Name` and `SMILES` only.
- **sample_submission.csv** — a correctly formatted submission example.

## Data fields

- **Molecule Name** — unique identifier for each molecule (e.g. `E-0011026`).
- **SMILES** — the molecular structure as a SMILES string.

The 9 target endpoints:

| Column | Endpoint | Units |
|---|---|---|
| `LogD` | Distribution coefficient (lipophilicity) | log units (already log scale) |
| `KSOL` | Kinetic solubility | µM |
| `MLM CLint` | Mouse liver microsomal intrinsic clearance | mL/min/kg |
| `HLM CLint` | Human liver microsomal intrinsic clearance | mL/min/kg |
| `Caco-2 Permeability Papp A>B` | Apparent permeability, apical→basolateral | 10⁻⁶ cm/s |
| `Caco-2 Permeability Efflux` | Caco-2 efflux ratio | ratio (unitless) |
| `MPPB` | Mouse plasma protein binding | % unbound |
| `MBPB` | Mouse brain protein binding | % unbound |
| `MGMB` | Mouse gastrocnemius muscle binding | % unbound |

## Important characteristics of this data

**The label matrix is very sparse.** Most molecules were measured for only one or two
endpoints; the remaining entries are missing (`NaN`). A molecule with an `HLM CLint` value
frequently has no `LogD`, `KSOL`, or Caco-2 measurement, and endpoints differ by an order of
magnitude in how many labels they have.

Two direct consequences:

- Dropping rows with any missing target would discard almost the entire dataset. Handle
  missingness explicitly — e.g. per-endpoint models, or a multi-task model with a **masked
  loss** that backpropagates only through observed entries.
- Because `MA-RAE` weights all 9 endpoints equally, the endpoints with the **fewest**
  measurements matter just as much as the well-measured ones. A model that is excellent on
  the two data-rich endpoints and poor on the sparse ones will score badly.

**Endpoint distributions are heavily right-skewed.** Clearance, solubility and permeability
values span orders of magnitude, which is precisely why the metric is computed in log space.
Training targets in a log-like space is usually far more effective than fitting raw values —
just remember to invert any transform before writing the submission, since the scorer expects
original units.

## Citation

ExpansionRx–OpenADMET Blind Challenge (2025–2026). OpenADMET & Expansion Therapeutics.
https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge
