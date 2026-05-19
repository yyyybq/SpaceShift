# Video Consistency v2

## Location

- Dataset: `video_consistency_thor_eval_v2/`
- QA file: `video_consistency_thor_eval_v2/qa.json`
- Group file: `video_consistency_thor_eval_v2/consistency_groups.json`
- Stats: `video_consistency_thor_eval_v2/dataset_stats.json`
- Selection report: `video_consistency_thor_eval_v2/selection_report.json`

## Metric Contract

- `1 group = 1 QA pair x 10 videos`
- `overall CV = mean(intra-group CV)`
- `MRA = global over all 10n responses`

## Benchmark Summary

| Metric | Value |
| --- | ---: |
| Total groups | 300 |
| Total videos | 3000 |
| Videos per group | 10 |
| Family split | 100 / 100 / 100 |
| Unique scenes | 63 |
| Unique atomic object labels | 51 |

## Distribution Notes

- Family split is exactly `100 / 100 / 100`.
- Motion families cover `rotation`, `spherical`, `around`, `approach`, and `passby`.
- `FloorPlan1` contributes `50` groups, but the full benchmark covers `63` scenes overall.
- `CoffeeTable` is capped at `13` groups.
- Validation status: `300 groups`, `issue_groups = 0`, `0` missing video paths, `0` symlink clip dirs.

## Top Scenes

| Scene | Groups |
| --- | ---: |
| FloorPlan1 | 50 |
| FloorPlan203 | 12 |
| FloorPlan209 | 12 |
| FloorPlan213 | 12 |
| FloorPlan215 | 12 |
| FloorPlan218 | 11 |
| FloorPlan230 | 10 |
| FloorPlan206 | 9 |

## Top Atomic Object Labels

| Label | Groups |
| --- | ---: |
| Chair | 21 |
| ArmChair | 20 |
| GarbageCan | 20 |
| Cabinet | 20 |
| HousePlant | 19 |
| Box | 19 |
| DiningTable | 18 |
| Statue | 17 |

## Analysis Figures

![Video QA distribution](../figures/analysis/video_v2/qa_distribution_thor.png)

![Video benchmark overview](../figures/analysis/video_v2/video_benchmark_overview.png)
