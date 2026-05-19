# Image Consistency v2

## Location

- Dataset: `image_consistency_thor_eval_v2/`
- QA file: `image_consistency_thor_eval_v2/qa.json`
- Group file: `image_consistency_thor_eval_v2/consistency_groups.json`
- Stats: `image_consistency_thor_eval_v2/dataset_stats.json`

## Benchmark Summary

| Metric | Value |
| --- | ---: |
| Total groups | 300 |
| Total images | 3000 |
| Images per group | 10 |
| Family split | 100 / 100 / 100 |
| Unique scenes | 77 |
| Unique anchor labels | 23 |

## Distribution Notes

- Engine is `thor` only.
- Question families are perfectly balanced across `camera_distance`, `pair_distance`, and `size`.
- Question types are perfectly balanced across `object_distance_to_camera`, `object_pair_distance_center`, and `object_dimensions`.
- Images are fully materialized under `image_consistency_thor_eval_v2/images/`.
- Validation status: `3000` image files, `0` symlinks, `0` missing materialized targets.
- Pair-distance validation: `1000 / 1000` images are now backed by source `frame_visibility.json` sidecars that confirm both anchors are co-visible in the selected frame.

## Top Scenes

| Scene | Count |
| --- | ---: |
| FloorPlan203 | 140 |
| FloorPlan209 | 130 |
| FloorPlan213 | 120 |
| FloorPlan218 | 120 |
| FloorPlan229 | 120 |
| FloorPlan230 | 120 |
| FloorPlan10 | 110 |
| FloorPlan18 | 110 |

## Top Anchor Labels

| Label | Count |
| --- | ---: |
| Box | 380 |
| Chair | 370 |
| CoffeeTable | 370 |
| DiningTable | 320 |
| ArmChair | 280 |
| Laptop | 260 |
| HousePlant | 210 |
| GarbageCan | 110 |

## Analysis Figures

![Image benchmark distributions](../figures/analysis/image_v2/image_benchmark_distributions.png)
