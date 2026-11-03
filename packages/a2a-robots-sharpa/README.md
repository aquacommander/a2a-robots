# a2a-robots-sharpa

[Sharpa Wave](https://github.com/sharpa-robotics) hands for [a2a-robots](../../README.md). This release has the MuJoCo backend; the Sharpa SDK backend for the real hands will be added here once the SDK is available.

## MuJoCo backend

`SharpaMujocoBackend` runs `dual_sharpa_wave.xml` from [sharpa-robotics/sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml) (Apache-2.0): two hands, each on a 6-DoF floating base, with no arm or body model.

- The hand pose is the palm frame (`<hand>_hand_C_MC`) in `world`. `move_to` solves the base joints for the target pose and drives them with the model's position actuators. A target outside the base joint range, or one the hand does not settle at, returns `unreachable`.
- `close` bends the flexion joints to `grasp_closure` (default 0.6) of their range and keeps abduction at zero; `open` straightens the fingers. `contact` is the normal force in newtons on each finger (and `palm`) from anything outside the hand. `held` means the thumb and another finger are both in contact. A close that touches something but ends without holding it returns `slipped`.
- The hand bodies have gravity compensation, as if mounted on an arm, so the base does not sag. Objects in the scene still fall.
- The simulation advances only while a skill runs, paced to wall-clock time unless `realtime=False`. Every control step (20 ms) checks the stop flag.

The meshes are about 24 MB, so they are not committed here. `fetch_model()` downloads the model at a pinned commit on first use, checks its digest, and caches it under `~/.cache/a2a-robots/`.

```bash
uv sync --all-packages --extra dev
uv run python packages/a2a-robots-sharpa/examples/serve_mujoco.py   # http://127.0.0.1:9100/
```

To add objects, edit the spec before compiling, as the example does:

```python
spec = mujoco.MjSpec.from_file(str(fetch_model()))
# spec.worldbody.add_geom(...) or add_body(...)
backend = SharpaMujocoBackend(load_model(spec))
```

## Tests

The unit tests use `tests/mini_dual_hand.xml`, a mesh-free model with the same joint and body names, so they need no download. `test_real_sharpa_model_moves_and_grasps` runs against the real model when `SHARPA_MODEL_DIR` is set:

```bash
uv run python -m a2a_robots_sharpa.model ~/.cache/a2a-robots/sharpa-wave
SHARPA_MODEL_DIR=~/.cache/a2a-robots/sharpa-wave uv run pytest packages/a2a-robots-sharpa
```
