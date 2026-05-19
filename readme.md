This repository keeps only two live benchmark datasets at the top level:

- `video_consistency_thor_eval_v2/`
- `image_consistency_thor_eval_v2/`

Historical datasets, scratch outputs, and reusable source assets belong under
`old/`. Temporary benchmark assembly directories should not remain at the
repository root after a cleanup pass.

The current downstream evaluation task lives in:

- `/nas2/edwin/lmms-eval/lmms_eval/tasks/sceneshift/video_consistency_production.yaml`
- `/nas2/edwin/lmms-eval/lmms_eval/tasks/sceneshift/video_consistency_production_utils.py`
