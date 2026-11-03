# Local replay resources

Inventory checked on **2026-11-03**. [replay_resources.json](replay_resources.json) groups the existing files by dataset, corpus, source run, parent bundle, baseline and experiment. It records their current locations and hashes; it is documentation, not a configuration file read by the launcher. All paths in that JSON are relative to the user's home directory.

## Where to look

| Resource | Current location under `~/` | Meaning |
|---|---|---|
| Public Jigsaw inputs | `nautilus/replay-inputs/{s68,jubias-f327226c}/runs/<source-run>/workspace/input/` | Two identical copies of train/test CSVs, sample submission and description |
| Private Jigsaw labels | `nautilus/replay-inputs/s68/mlebench/jigsaw-unintended-bias-in-toxicity-classification/prepared/private/` | Grading data; the six recorded Mac replays did not configure private grading |
| Replay paper corpus | `nautilus/replay-inputs/s68/paper_corpus/` | Shared 38,273-paper corpus, fingerprint `5cd1d5e7afd0` |
| Downloaded source runs | `nautilus/replay-inputs/{s68,jubias-f327226c}/runs/` | Partial historical recovery artifacts for the two parents |
| Download/selection provenance | Each download folder's `download_manifest.json`; also `jubias-f327226c/selection.json` and `candidate_inventory.jsonl` | Original manifests and random-selection record |
| Mac baselines | `Documents/project/python/MLEvolve/mac_baselines/` | Two completed calibrated parents and one retained failed calibration |
| Replay results | `Documents/project/python/MLEvolve/runs_mac_mps/` | Six experiments, each with completed A/B/C/D arms |

`s68` is the name of the first download batch, not ownership by seed 68. The corpus and data there also support seeds 67 and 69 and the second parent. Only `paper_corpus/` is needed for paper retrieval; replay additionally needs the source run, its public inputs and its completed Mac baseline. The `s68/mlebench/` tree contains the downloaded private test file, not a complete public/private dataset tree.

## Parent bundles

| Logical bundle | Historical source | Completed baseline | Result family, seeds 67/68/69 |
|---|---|---|---|
| `jubias-36845c7c` | `20260914_071735_jubias-base-gpt56sol-s60`, parent `36845c7c…` | `jubias-parent5-s68-torch2141` | `replay_jubias-noise-parent5-mac-sNN` |
| `jubias-f327226c` | `20260912_081234_jubias-base-gpt56sol-s57`, parent `f327226c…` | `jubias-f327226c-torch2141` | `replay_jubias-f327226c-mac-sNN` |

These bundle names are catalog entries, not newly created directories. The failed `jubias-parent5-s68` calibration remains available for diagnosis and must not be used as the replay baseline. The existing second-parent launcher remains at `~/nautilus/replay-inputs/jubias-f327226c/run_seed.sh`; it refuses to overwrite completed runs. Entry points and Mac usage are described in [replay_macos.md](replay_macos.md).

## What remains in place

- All original directories, manifests, launchers, baselines and results retain their paths. Baseline identity includes resolved source paths, while experiment identity includes corpus paths and the baseline manifest hash. Renaming folders or rewriting old manifests would require a separate migration.
- The two public-input copies contain the same 847,325,974 bytes, but each source has a different validation contract (source seed 60 versus 57). Keep those contracts separate. No deduplication or physical disk-space estimate was performed.
- Downloaded source runs omit original checkpoint weights; their recovery artifacts support Mac recalibration, not arbitrary restoration of the original GPU checkpoint.
- `~/Documents/project/python/Agentic_Knowledge_Base/output/paper_corpus/` currently has **12,800 papers**, fingerprint `35d01b8b78ae`. It is a different corpus and must not silently replace the replay corpus.

Inventory verification checked all **210 files** listed by the two download manifests against size and SHA256, the two public-input copies, parent/baseline/corpus identities for all six experiments, and completion status for all 24 arms. The old manifests use different file-map bases; those bases are explicit in the JSON. This check did not rerun training, scoring or a full checkpoint audit. Update this static catalog when adding resources; preserve the original experiment manifests as the historical record.
