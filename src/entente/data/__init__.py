"""Turn human game records into the sharded rows behavioral cloning trains on.

Paquette JSONL games (:mod:`entente.data.paquette`) -> filter to standard-map
completed games and reshape to per-(movement-phase, power) rows in sharded
Parquet (:mod:`entente.data.preprocess`) -> a torch ``Dataset`` streaming those
shards (:mod:`entente.data.dataset`). :mod:`entente.data.synthetic` generates
Paquette-schema games from engine self-play, which gives the pipeline a
teacher when the human dump is unavailable and produces the committed test
fixture.

Only :mod:`entente.data.dataset` imports torch; the reader, synthesizer, and
preprocessor are torch-free (``preprocess`` uses pyarrow from the ``ml`` group).
"""
