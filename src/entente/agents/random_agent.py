"""A seeded uniform-random agent -- the baseline every other agent is measured against.

Movement: each unit independently picks one of its legal orders (holds,
moves, supports, convoys -- plus the ruleset's Layer-2 orders where it
ships them) uniformly. Retreats: each dislodged unit picks
uniformly among its retreat moves and disbanding. Adjustments: a
build-entitled power fills each build slot with a uniformly chosen still-free
site or a waive; a removing power picks a uniform subset of its units.

Determinism contract: the sheet is a pure function of (seed, power, phase
tag, position) -- the RNG is rebuilt from those on every call, so replaying a
game or running powers in any order reproduces identical sheets. Seeding
uses ``random.Random(str)``, which hashes the string with SHA-512 -- stable
across platforms and Python versions (CPython guarantee).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from entente.engine.orders import Build, Disband, Waive
from entente.engine.phase import PhaseKind

if TYPE_CHECKING:
    from collections.abc import Sequence

    from entente.engine.game import Game
    from entente.engine.orders import GameOrder


class RandomAgent:
    """Uniform random play over the legal sheet, from a seed.

    A movement sheet draws from ``game.legal_orders`` together with
    ``game.legal_layer2_orders``, so a variant's own order types are in the
    draw wherever the ruleset ships them.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def _rng(self, game: Game, power: str) -> random.Random:
        """Derive this call's stream from the seed, the power and the position."""
        return random.Random(f"{self.seed}:{power}:{game.phase}:{game.state.units}")

    def orders(self, game: Game, power: str) -> Sequence[GameOrder]:
        """Return ``power``'s uniformly drawn sheet for ``game``'s current phase."""
        rng = self._rng(game, power)
        legal: tuple[GameOrder, ...] = game.legal_orders(power)
        if game.phase_kind is PhaseKind.MOVEMENT:
            legal = legal + game.legal_layer2_orders(power)
        if not legal:
            return []
        kind = game.phase_kind
        if kind is PhaseKind.MOVEMENT:
            return self._per_unit_choice(rng, legal)
        if kind is PhaseKind.RETREAT:
            return self._per_unit_choice(rng, legal)
        return self._adjustment_choice(rng, game, power, legal)

    def _per_unit_choice(self, rng: random.Random, legal: Sequence[GameOrder]) -> list[GameOrder]:
        """One uniform pick per unit (orders grouped by the ordering unit's location)."""
        by_unit: dict[str, list[GameOrder]] = {}
        for order in legal:
            assert not isinstance(order, Waive)  # movement/retreat orders name a unit
            by_unit.setdefault(str(order.location), []).append(order)
        return [rng.choice(options) for _, options in sorted(by_unit.items())]

    def _adjustment_choice(
        self, rng: random.Random, game: Game, power: str, legal: Sequence[GameOrder]
    ) -> list[GameOrder]:
        """Draw a removal subset, or fill each build slot with a free site or a waive."""
        builds = [order for order in legal if isinstance(order, Build)]
        disbands = [order for order in legal if isinstance(order, Disband)]
        if disbands:
            count = min(-game.build_quota(power), len(disbands))
            return list(rng.sample(disbands, count))
        sheet: list[GameOrder] = []
        used: set[str] = set()
        for _ in range(game.build_quota(power)):
            options: list[GameOrder] = [b for b in builds if b.location.province not in used]
            options.append(Waive())
            choice = rng.choice(options)
            sheet.append(choice)
            if isinstance(choice, Build):
                used.add(choice.location.province)
        return sheet
