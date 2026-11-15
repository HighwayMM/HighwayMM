# Training launcher (STEP 1)

Native Windows, 3x RTX 3090, torch 2.5.1+cu121, **gloo** DDP (no NCCL).

| file | purpose |
|---|---|
| `launch.ps1` | entry point: `-Gpus N [-Visible "0,1"] <script> <args>` |
| `spawn.py` | torchrun replacement (torchrun's static rendezvous crashes on Windows: "built without libuv support") |
| `ddp.py` | setup/cleanup, seeding, reductions, `maybe_no_sync` |
| `runlog.py` | mandatory run folder + `experiments\runs_index.csv` (status running → completed/failed/aborted); `mark_run(id,'invalid',reason)` |
| `trainer.py` | generic trainer: bf16 AMP, grad accumulation with `no_sync`, freeze prefixes, warmup+cosine, best/last ckpt, exact resume, preds, `curves.png` |
| `main.py` | `--config X.yaml --set a.b=v ... --tag t`; loads `tasks/<task>.py: build(cfg, ctx)` |
| `jobrunner.py` | one worker per GPU for parallel single-GPU jobs (`--cmd "... seed={seed}" --grid seed=0,1,2`) |
| `bench.py` | gradient-equality / resume / throughput sanity benchmark |

```powershell
powershell -File code\train\launch.ps1 -Gpus 3 code\train\main.py --config code\train\configs\synthetic.yaml
python code\train\jobrunner.py --gpus 0,1,2 --cmd "code\train\main.py --config <cfg> --set seed={seed}" --grid seed=0,1,2
```

Add a task: `code\train\tasks\<name>.py` with `build(cfg, ctx) -> dict(model, train_ds, val_ds, [loss_fn, eval_fn, monitor, mode, collate_fn])`.
Datasets return `(x, y, id)`; `eval_fn(model, loader, ctx, amp_dtype) -> (metrics_dict_reduced, per_sample_preds)`.

Harmless noise: `socket.cpp ... failed to connect to [DESKTOP-...]` warnings at startup (gloo retries before the store is up).
Benchmark results and guidance: `analysis\step1_ddp_benchmark.md`.
