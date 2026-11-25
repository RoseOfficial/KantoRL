# Model cards

Two policies are released. Both are PPO `MultiInputActorCriticPolicy` checkpoints
on the seven-component observation, trained from the same `bulba_start` save
state. Neither completes the game.

## `red_99M_s43.zip` — deepest policy

**Provenance.** Final checkpoint of
`t2_levelmax_h60_lr5e5_100M_h60g_s43` at 99.0M steps, seed 43. That run resumed
from the 90M policy at 5e-5; the 90M policy in turn resumed from the 78.8M
policy. The 100M final collapsed to 5/10 Boulder, so 99.0M was selected by
checkpoint sweep.

**Configuration.**

| setting | value |
|---|---|
| observation | `screens`, `health`, `level`, `badges`, `events`, `map`, `recent_actions` |
| screens | 3 stacked 72x80 grayscale frames |
| map | 48x48 visited bitmap, no player marker |
| action frequency | 24 emulated frames |
| horizon | 61,440 steps |
| reward | default; `level_weight` 4.0, `level_source` max, `explore_weight` 0.3, `reward_scale` 0.5 |
| PPO | 32 envs, 128 steps/env, `gamma` 0.995, `target_kl` 0.02 |
| learning rate | 5e-5 (continuation) |
| curriculum | off |

**Measured.** 10 stochastic seeds, 61,440 steps, `bulba_start`, corrected party
reads:

| milestone | reach | median steps |
|---|---:|---:|
| Boulder Badge | 10/10 | 7,182 |
| Route 3 | 10/10 | 7,440 |
| Route 3 trainer 5 | 9/10 | 18,911 |
| Mt. Moon 1 trainer 2 | 10/10 | 21,928 |
| Mt. Moon 1 trainer 5 | 9/10 | 16,759 |
| Mt. Moon 1 trainer 6 | 8/10 | 30,826 |
| Mt. Moon 3 trainer 2 | 9/10 | 28,482 |
| Mt. Moon B2F | 10/10 | 24,587 |
| Exit Super Nerd, Cerulean | 0/10 | -- |

Full report: `eval_990M_10x60k_fixed/`, inside `eval_reports.zip` on the
release.

**Caveats.** Trained before the party-memory fix; the levels and HP fed to PPO
during that run were read with the wrong stride and byte order. Evaluation
under the corrected reads still scores 10/10 Boulder, so the artifact is valid
but its training signal was partly corrupted. The checkpoint is a
path-dependent development result: a clean from-scratch run of the same rules
failed before Brock. Do not cite it as a reproducible sample-efficiency result.

**Use.** Deepest known warm start for a continuation. `red_99M_s43_vecnormalize.pkl`
is the 100M run's final normalization state (the per-checkpoint file was not
saved). Observation normalization is off, so this only matters for reward
normalization on resume.

## `red_gamma_104M_s43.zip` — speed/reliability policy

**Provenance.** 104M final of `t2_levelmax_h60_lr2e5_g999`, a 5M continuation
of the 99M policy at 2e-5 with `gamma` raised from 0.995 to 0.999. Observation,
reward, horizon and seed are unchanged.

**Configuration.** Same as the 99M policy except `gamma=0.999` and
`learning_rate=2e-5`, trained for 104M total.

**Measured.** Same protocol:

| milestone | reach | median steps |
|---|---:|---:|
| Boulder Badge | 10/10 | **5,160** |
| Route 3 trainers 0-3, 6-7 | 9-10/10 | 8.0k-11.8k |
| Route 3 trainer 5 | 3/10 | 12,634 |
| Mt. Moon 1 trainer 1 / 2 / 4 | 8/10, 10/10, 10/10 | 39,940 / 25,846 / 14,245 |
| Mt. Moon 1 trainer 5 | 7/10 | 25,554 |
| Mt. Moon 1 trainer 6 | 0/10 | -- |
| Mt. Moon 3 trainer 2 | **10/10** | 31,839 |
| Mt. Moon B2F | 10/10 | 28,590 |
| Exit Super Nerd, Cerulean | 0/10 | -- |

Full report: `eval_104M_10x60k/`, inside `eval_reports.zip` on the release.

**Caveats.** The 99M policy is deeper in Mt. Moon 1 and Route 3 trainer 5. This
policy trades that depth for a faster badge and a more reliable Mt. Moon 3
Trainer 2. It is the clearest positive method result in the project: a single
`gamma` change, no new observation and no new reward term.

**Use.** Best base when early speed and Mt. Moon 3 reliability matter. It is a
warm-start artifact on the same corrupted-read lineage as the 99M policy.

## Loading

```python
from kantorl.checkpoints import load_model

model = load_model("red_99M_s43.zip")
```

`load_model` reads the policy class and observation contract from the archive,
so the checkpoint loads without rebuilding the tier by hand.

## License

The checkpoints are distributed as release assets and are covered by the
repository license: all rights reserved. Pokémon Red and its assets belong to
Nintendo, Game Freak and Creatures. No ROM is distributed here.
