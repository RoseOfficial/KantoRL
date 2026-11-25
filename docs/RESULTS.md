# Results: the early game, measured

This is the short version of what KantoRL measured. The chronological lab log,
including every harness bug and dead end, is in
[EXPERIMENTS.md](EXPERIMENTS.md). Checkpoint provenance and evaluation commands
are in [MODELS.md](MODELS.md).

## Summary

KantoRL trains PPO on Pokémon Red from a fixed Oak's Lab save state. There is
no stage manager, no scripted HM use, no route curriculum and no mid-episode
savestate resets. Episodes are capped at 61,440 environment steps.

| result | policy | number |
|---|---|---|
| Deepest policy | 99M, seed 43 | Boulder 10/10, Route 3 10/10, Mt. Moon B2F 10/10 |
| Fastest reliable badge | gamma 104M | Boulder 10/10, median 5,160 steps |
| Frontier reached | both | Mt. Moon B2F; Exit Super Nerd 0/10 |
| Clean from-scratch confirmation | 1e-4 schedule | **failed** (0/3 Boulder at 24.1M) |

The full game was not completed. The current policies reach Mt. Moon B2F in
every episode and have never beaten the Exit Super Nerd or entered Cerulean.

## Protocol

- Pokémon Red (UE), 24 emulated frames per action, button held for 8.
- Start state `bulba_start`: Oak's Lab, level-6 Bulbasaur, 11 event flags
  already set. Training and evaluation use the same state.
- Observation: three stacked 72x80 grayscale frames, party HP, Fourier-encoded
  party levels, badge flags, the 2,560-bit event field, a 48x48 visited-tile
  map, and the last three actions.
- Reward: +4 per new event flag, +5 per badge, +1 per new map, +0.015 per new
  `(map, x, y)` cell, +0.05 for a first interaction at a coordinate, and a
  potential-based level reward on the strongest party member (weight 4.0:
  +2.0 below level 15, +0.5 above). Healing and stuck penalties are off because
  both were farmable.
- PPO with 32 environments, 128 steps per environment per update, `gamma=0.995`,
  `target_kl=0.02`, learning rate 1e-4 until 78M then 5e-5, 61,440-step horizon.
- Evaluation is stochastic, 10 seeds, 61,440 steps, from the same save state.
  Milestones are read from RAM **for evaluation only** and never fed to the
  policy, so a pixels-only tier and a RAM-rich tier are measured by the same
  ruler.
- Seeds 0-9 were the development set. Seeds 10-19 were reserved for the final
  fixed-ruleset measurement and were never spent, because no checkpoint passed
  the pre-registered validation gate.

## Final results

Both released policies, 10 stochastic seeds, 61,440 steps. Medians are
environment steps to first reach the milestone.

| milestone | 99M (deepest) | gamma 104M (speed) |
|---|---:|---:|
| Boulder Badge | 10/10, 7,182 | **10/10, 5,160** |
| Route 3 trainers 0-4, 6-7 | 10/10 | 10/10 (t2 9/10; t4 unmeasured) |
| Route 3 trainer 5 | **9/10, 18,911** | 3/10, 12,634 |
| Mt. Moon 1 trainers 1, 2, 4 | 10/10 | 8/10, 10/10, 10/10 |
| Mt. Moon 1 trainer 5 | 9/10, 16,759 | 7/10, 25,554 |
| Mt. Moon 1 trainer 6 | **8/10, 30,826** | 0/10 |
| Mt. Moon 3 trainer 2 | 9/10, 28,482 | **10/10, 31,839** |
| Mt. Moon B2F | **10/10, 24,587** | 10/10, 28,590 |
| Exit Super Nerd | 0/10 | 0/10 |
| Cerulean City | 0/10 | 0/10 |

The 99M policy is deeper: it clears more of the Mt. Moon 1 trainer line and
reaches the B2F exit more reliably. The gamma policy is faster and more
reliable on the badge and Mt. Moon 3 Trainer 2, but loses Mt. Moon 1 depth
(trainer 6 drops from 8/10 to 0/10). Neither reaches Trainer 3, the Exit Super
Nerd, or Cerulean.

The 99M checkpoint was trained before a party-memory fix: levels and HP were
read with the wrong struct stride and byte order, and could report fake level-89
party members. It still scores 10/10 under the corrected reads, so the
evaluation above is valid; the training signal was not.

## The clean from-scratch run failed

The development lineage above is not a clean sample-efficiency claim. A
pre-registered 100M from-scratch run (same rules, seed 43, 1e-4 until 78M) was
stopped at 24.1M because it never reliably beat Brock:

| checkpoint | Boulder (validation seeds 0-2) | Viridian Forest | Pewter |
|---|---:|---:|---:|
| 5.33M badge snapshot | 0/3 | -- | 0/3 |
| 10M | 0/3 | 2/3 | 0/3 |
| 20M | 0/3 | 3/3 | 0/3 |
| 24.1M | 0/3 | -- | -- |

The 5.33M first badge was a training-only transient; the saved policy does not
reproduce it from a fresh start. At 24.1M the training window was recovering
(`badges_mean` 0.03 -> 0.31), but the checkpoint did not reproduce it. The
5e-5 drop at 78M was never reached.

The selection rule was fixed before the run (`scripts/clean_select.py`): rank
the 10M checkpoints by Boulder reach, then Mt. Moon 3 Trainer 2, then B2F,
then Boulder median. No checkpoint passed the first criterion, so the held-out
seeds 10-19 were never spent.

This is the most important negative result in the project. It says the 99M
policy is a path-dependent development result, not something the fixed recipe
reliably produces from zero. Any claim of the form "PPO solves early Pokémon
Red in 99M steps" is not supported by this repository.

## Negative results

Every 5M warm-start probe from the 99M policy was evaluated on the same 10
seeds. All regressed except gamma.

| probe | change | Boulder | Mt. Moon 3 t2 | Mt. Moon B2F |
|---|---|---:|---:|---:|
| 99M baseline | -- | **10/10** | **9/10** | **10/10** |
| gamma | 0.995 -> 0.999 | **10/10 (5,160)** | **10/10** | 10/10 |
| local visited map | +15x15 window | 9/10 | 1/10 | 7/10 |
| reward rebalance | level 1.0, explore 1.0 | 8/10 | 8/10 | 8/10 |
| player marker | map cell = 128 | 10/10 | 5/10 | 2/10 |
| raw tile IDs | +15x15 tile window | 6/10 | 3/10 | 6/10 |

Two patterns are consistent across the probes:

1. **The 99M policy is a sharp local optimum.** A 5M fine-tune at 2e-5 to 5e-5
   moves the shared encoder enough to destroy depth, while the new branch
   learns a small readout of mostly-random features.
2. **More local detail is not the missing ingredient.** The local visited map,
   raw tile IDs and player marker all add spatial information and all make the
   indoor result worse. The coarse 48x48 map is load-bearing, but its resolution
   is not the only limit.

Gamma 0.999 is the one positive method change: it shortens the badge median by
28% and makes Mt. Moon 3 Trainer 2 10/10, at the cost of Mt. Moon 1 depth. It
changes no observation and no reward.

## What the project establishes

- **The map is the single biggest observation contributor.** In a matched
  20M-step ablation, pixels-only reached Viridian City 0/10; pixels plus the
  map reached 5/5, matching the full tier. Without the map, the RAM fields
  alone reached Viridian 2/5 but never Route 2 or Viridian Forest.
- **The 48x48 map is an indoor ceiling.** One cell is roughly 9x9 tiles, and
  the fossil room is smaller than one cell. Minimal local maps do not fix it.
- **The current recipe is not reproducible from scratch.** The clean run fails
  before Brock, so the 99M result is a development lineage, not a recipe.
- **Warm-start fine-tuning is not a safe experimental method here.** Four
  independent 5M probes regressed the base policy. The one positive change
  (gamma) was also the smallest intervention.
- **Reward shaping is not the bottleneck at this stage.** Rebalancing level and
  exploration reward degraded the core policy rather than buying exploration.

## Threats to validity

- The headline policy is one seed (43) selected by checkpoint sweep. The clean
  replication failed, so seed sensitivity is unresolved.
- The 99M evaluation uses 10 seeds; reach rates near 8/10 or 9/10 have wide
  error bars.
- The 99M checkpoint predates the party-memory fix. Evaluation is correct;
  the training run saw corrupted levels and HP.
- All numbers are from one machine (8-core/16-thread CPU), PyBoy 2.9.0, with
  `fast_mode=True`. That mode changes emulator timing and is not step-comparable
  with pre-`fast_mode` runs.
- Episodes end at 61,440 steps. Deeper milestones are censored, not proven
  unreachable.
- Milestones are curated from RAM. They measure specific flags, not general
  game competence; a policy can pass a flag while playing badly.

## Reproducing the evaluation

The released checkpoints are described in [MODELS.md](MODELS.md). Download an
asset, put the ROM and the `bulba_start` state outside the repository, then:

```powershell
kantorl milestones red_99M_s43.zip "C:/roms/Pokemon Red.gb" `
    --seeds 0 1 2 3 4 5 6 7 8 9 --budget 61440 --stochastic `
    --save-state "C:/roms/states/bulba_start.state" `
    --out runs/eval/red_99M_repro
```

Use the same command with `red_gamma_104M_s43.zip` for the speed policy. The
ROM and save state are not distributed with this repository.

## Artifacts

| file | what it is |
|---|---|
| `red_99M_s43.zip` (release asset) | deepest policy |
| `red_gamma_104M_s43.zip` (release asset) | speed/reliability policy |
| `eval_reports.zip` (release asset) | full 99M and gamma milestone reports |
| `docs/EXPERIMENTS.md` | chronological log and harness bugs |

SHA-256:

```
dee69e62fd7e9a176b814499ab8429deaf7e21f8d3b421306245cffa40ea4dd3  red_99M_s43.zip
2a87831ba5ce963335887de962aba10f307f9b90040d7662313c50c2d93f520f  red_gamma_104M_s43.zip
```
