"""Provide the neural policy/value stack used for behavioral cloning.

A location-token transformer over the board (:mod:`entente.nn.features`,
:mod:`entente.nn.encoder`) with an autoregressive per-unit order head and a
7-power value head (:mod:`entente.nn.heads`), assembled and checkpointed by
:mod:`entente.nn.model`. The policy head decodes over a precomputed global order
vocabulary (:mod:`entente.nn.order_vocab`) with engine-exact legality masking.

Only :mod:`entente.nn.order_vocab` is torch-free (it drives the engine's own
legal-order enumeration); everything else imports torch and is part of the
``ml`` dependency group, exercised only by ``ml``-marked tests.
"""
