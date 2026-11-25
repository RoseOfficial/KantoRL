# KantoRL v0.1.0 — final development snapshot

This release freezes the KantoRL development program. It is a research
artifact, not a full-game solve.

## Headline

| result | policy | number |
|---|---|---|
| Deepest policy | 99M, seed 43 | Boulder 10/10, Route 3 10/10, Mt. Moon B2F 10/10 |
| Fastest reliable badge | gamma 104M | Boulder 10/10, median 5,160 steps |
| Frontier | both | Exit Super Nerd / Cerulean 0/10 |
| Clean from-scratch confirmation | 1e-4 schedule | failed at 24.1M (0/3 Boulder) |

The 99M policy is the deepest development result. The gamma-0.999 104M policy
is the speed/reliability alternative. Neither reaches the Exit Super Nerd.
The clean from-scratch run failed before Brock, so the 99M result is
path-dependent and must not be cited as a reproducible sample-efficiency
claim.

## Assets

| asset | size | SHA-256 |
|---|---:|---|
| `red_99M_s43.zip` | 33,229,394 B | `dee69e62fd7e9a176b814499ab8429deaf7e21f8d3b421306245cffa40ea4dd3` |
| `red_gamma_104M_s43.zip` | 33,229,403 B | `2a87831ba5ce963335887de962aba10f307f9b90040d7662313c50c2d93f520f` |
| `red_99M_s43_vecnormalize.pkl` | 792,943 B | `919f0c671f3ee4f62a64880a1cbb2854f22869f4899ba4b5b28b77377b32786d` |
| `red_gamma_104M_s43_vecnormalize.pkl` | 792,943 B | `c8d2a7b10921b0029e9d1c34ea694c433303ef95f2a4cd3a51703e928276f9b6` |
| `eval_reports.zip` | 26,085 B | `fae7f68a7e80e7acb0664555041901751ee3a074ca0d17992c75042e208e3c2d` |

The `.pkl` files are the reward-normalization states from the parent runs.
Observation normalization is off. They are only needed when resuming training;
evaluation loads the `.zip` alone.

## What is in the repository

- A minimal Gymnasium environment for Pokémon Red (`src/kantorl/`).
- PPO training with composable observation producers, reward protocols and
  named YAML tiers.
- First-step milestone evaluation from RAM, used only as a ruler and never fed
  to the policy.
- 167 tests covering memory reads, rewards, observation encoders, tiers,
  checkpoint loading and milestone selection.
- The chronological experiment log, including the harness bugs:
  [docs/EXPERIMENTS.md](https://github.com/RoseOfficial/KantoRL/blob/main/docs/EXPERIMENTS.md).
- The narrative summary and threats to validity:
  [docs/RESULTS.md](https://github.com/RoseOfficial/KantoRL/blob/main/docs/RESULTS.md).
- Checkpoint configuration and provenance:
  [docs/MODELS.md](https://github.com/RoseOfficial/KantoRL/blob/main/docs/MODELS.md).

## Verify a checkpoint

Install the package, point at a ROM and the `bulba_start` save state, then run
the same evaluation used for the release numbers:

```powershell
kantorl milestones red_99M_s43.zip "C:/roms/Pokemon Red.gb" `
    --seeds 0 1 2 3 4 5 6 7 8 9 --budget 61440 --stochastic `
    --save-state "C:/roms/states/bulba_start.state" `
    --out runs/eval/red_99M_repro
```

Boulder should reach 10/10 at a median near 7,182 steps (stochastic evaluation
will vary slightly across machines and seeds). The ROM and save state are not
distributed.

## Known limitations

- Not a full-game result: Trainer 3, the Exit Super Nerd and Cerulean are 0/10.
- The 99M checkpoint was trained before the party-memory fix; evaluation is
  corrected, the original training signal was not.
- Single-seed lineage, selected by checkpoint sweep, and not reproduced from
  scratch.
- Reach rates of 8/10 and 9/10 have wide error bars.
- `fast_mode=True` changes emulator timing; these numbers are not comparable
  with earlier `fast_mode=False` runs.
- Episode cap 61,440 steps; deeper milestones are censored, not proven
  unreachable.

## License

All rights reserved. See
[LICENSE](https://github.com/RoseOfficial/KantoRL/blob/main/LICENSE).
Pokémon Red and its assets belong to Nintendo, Game Freak and Creatures. No
ROM is included.
