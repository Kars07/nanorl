# Modal execution

The executable app lives in `modal_apps/` so this directory does not shadow the installed
`modal` package. These modules expose that same implementation, not a simulated backend.

```powershell
uv run modal run modal_apps/self_hosted_rollout.py --mode train-repo-repair
uv run modal run modal_apps/self_hosted_rollout.py --mode eval-repo-checkpoint
```
