# Benchmark Heat Exchanger Network Instances

This directory contains digitized literature benchmark problem instances used to evaluate PinchPy's deterministic analysis, network synthesis engine, and surrogate/guardrail pipelines.

## Collections

### 1. Furman & Sahinidis (2004)
Located in [`furman_sahinidis/dat_files/`](./furman_sahinidis/dat_files/):
- **26 problem files** (representing **25 unique heat integration problems**; `9sp-al1` and `10sp-la1` share identical process streams and differ only in utility cost specifications).
- Instances range from 4 streams (`4sp1`) to 37 streams (`37sp-yfyv`).
- Individual primary literature sources (Lee 1970, Masso 1969, Ciric 1989, Gundersen 1990, Colberg 1990, Shenoy 1995, etc.) are cataloged in [`furman_sahinidis/references.txt`](./furman_sahinidis/references.txt).

### 2. Chen, Grossmann, & Miller (2015)
Located in [`chen_grossmann_miller/dat_files/`](./chen_grossmann_miller/dat_files/):
- **10 problem instances**:
  - `balanced5`, `balanced8`, `balanced10`, `balanced12`, `balanced15`
  - `unbalanced5`, `unbalanced10`, `unbalanced15`, `unbalanced17`, `unbalanced20`
- Instances range from 10 streams (5 hot, 5 cold) to 40 streams (20 hot, 20 cold).
- Originally investigated in:
  > Chen, C., Grossmann, I. E., & Miller, D. C. (2015). *MINLP models for heat exchanger network synthesis under multiperiod operation.* Computers & Chemical Engineering, 82, 68–83.

## Source & Attribution

All `.dat` benchmark files in this repository were compiled and formatted into standard `.dat` syntax by Dimitrios Letsios, Georgia Kouyialis, and Ruth Misener (Imperial College London), made publicly available under the **Apache License 2.0** at:
- Repository: [`TYT0710/min_matches_heuristics`](https://github.com/TYT0710/min_matches_heuristics) (also mirrored at [`cog-imperial/min_matches_heuristics`](https://github.com/cog-imperial/min_matches_heuristics)).
- Paper:
  > Letsios, D., Kouyialis, G., & Misener, R. (2018). *Heuristics with guarantees for heat exchanger network synthesis.* Computers & Chemical Engineering, 113, 197–217.

## Benchmark Format

Files follow the standard `.dat` layout:
- `DTmin <value>`: Minimum approach temperature (°C)
- `HSn <Tin> <Tout> <CP>`: Hot process stream (supply T, target T, heat capacity flow rate in kW/°C)
- `CSn <Tin> <Tout> <CP>`: Cold process stream (supply T, target T, heat capacity flow rate in kW/°C)
- `HUn` / `CUn`: Utility streams (temperatures and unit costs)
