"""KL-regularized equilibrium search -- this project's centrepiece method (docs/pikl.tex).

Three torch-free layers: :mod:`entente.search.pikl` (the anchored-hedge update
over abstract candidate sets), :mod:`entente.search.candidates` (top-k order-set
sampling from any policy), :mod:`entente.search.rollout` (batched 1-ply
joint-profile evaluation against a pluggable value function). The torch-backed
glue lives in :mod:`entente.agents.pikl_agent`.
"""
