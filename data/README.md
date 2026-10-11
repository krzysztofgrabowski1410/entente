# `data/` -- datasets and derived artifacts (git-ignored)

This directory holds large, git-ignored artifacts: the Paquette human-games dump
and the Parquet shards preprocessed from it. Only this README is committed
(via a `.gitignore` exception). The tree is **shared across git worktrees** by
absolute path -- keep the download in the main checkout's `data/` so every
worktree reads the same file:

```
C:\Users\krisi\repository\diplomacy\data\
```

## The Paquette no-press dataset

Behavioral cloning trains on the Paquette et al. (2019) no-press
game dump -- **156,468 games**, moves only, ~2.24 GB as a JSONL zip. The reader
(`entente.data.paquette`) and preprocessor (`entente.data.preprocess`) consume
it; see `docs/setup.md` for the exact download and preprocessing commands and the
recorded SHA-256.

### Usage restriction (important)

The dataset's terms permit research use but **prohibit using it to power bots on
Diplomacy websites without the players' consent**. This
project uses the data only to train and evaluate agents offline, in tournaments
against other agents -- never to operate a bot against humans on any website. Any
release of trained weights must repeat this restriction.

## Synthetic fallback

When the dump is unavailable, `entente.data.synthetic` generates games in the
same schema from engine self-play, with a greedy teacher. These are
generated facts of play, MIT-licensed like the rest of the project, and are what
the committed test fixture (`tests/data/fixtures/synthetic_games.jsonl`) and the
smoke-training run use. They verify the pipeline and back the smoke curve; they
are **not** human games and no behavioral-cloning quality claim rests on them.

## The Peskov et al. (2020) dialogue-deception corpus

The negotiation prototype's dialogue fine-tune trains on **"It Takes Two to Lie:
One to Lie, and One to
Listen"** (Peskov, Cheng, Elgohary, Barrow, Danescu-Niculescu-Mizil,
Boyd-Graber; ACL 2020) -- 17,289 messages across 246 real Diplomacy-game
dialogues, each sender-labeled truthful/deceptive and receiver-labeled
perceived-truthful/deceptive/unannotated. Reader:
`entente.negotiation.data.peskov`; preprocessing to instruction-tuning pairs:
`entente.negotiation.data.instructions`.

```sh
mkdir -p data/peskov
curl --ssl-no-revoke -L -o data/peskov/train.jsonl \
  "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/master/data/train.jsonl"
curl --ssl-no-revoke -L -o data/peskov/validation.jsonl \
  "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/master/data/validation.jsonl"
curl --ssl-no-revoke -L -o data/peskov/test.jsonl \
  "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/master/data/test.jsonl"
```

- **SHA-256** of the files as downloaded:
  - `train.jsonl` (2,472,563 bytes): `833d8ef26561bbd1921e9f5e730b2ebea0a59cec24f7082bfdcacb8b6cf30792`
  - `validation.jsonl` (245,870 bytes): `6b2ca19991c7596693ea61b34b1be2d6ada3fcf6710472edd4afed841556e571`
  - `test.jsonl` (490,273 bytes): `2061ee7f040d6d8016760cb146d7821fef57cfbb4ced868f233e63854257c050`
- **License: CC-BY 4.0.** Unlike the Paquette dump's bespoke research-only
  terms, CC-BY permits redistribution and derivatives with attribution -- the
  committed test fixture (`tests/negotiation/fixtures/peskov_sample.jsonl`,
  two full real conversations -- already this short in the original, nothing
  trimmed) is therefore real corpus data, not a synthetic stand-in. **Any use
  must cite:**

  ```bibtex
  @inproceedings{Peskov:Cheng:Elgohary:Barrow:Danescu-Niculescu-Mizil:Boyd-Graber-2020,
    Title = {It Takes Two to Lie: One to Lie and One to Listen},
    Author = {Denis Peskov and Benny Cheng and Ahmed Elgohary and Joe Barrow and Cristian Danescu-Niculescu-Mizil and Jordan Boyd-Graber},
    Booktitle = {Association for Computational Linguistics},
    Year = {2020},
  }
  ```
- Source repository: <https://github.com/DenisPeskoff/2020_acl_diplomacy>
  (data files only -- this project's reader is an independent implementation
  from the format documented in that repo's `data/README.md`, not a copy of
  its `diplomacy/readers/` code -- this project's own code is written from
  specifications, never copied).
