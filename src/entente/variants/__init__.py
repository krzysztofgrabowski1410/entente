"""Variant rulesets: each subpackage instantiates the calculus for one rulebook.

A variant is data plus a ``Ruleset`` object (docs/adjudication.tex Sec. 7).
Variants depend on the engine's public surface; the engine never imports a
variant, so the dependency runs one way. ``entente/__init__`` imports the
subpackages so they self-register: the ruleset in
``entente.engine.rules.RULESETS``, the game hooks in
``entente.engine.game_hooks.GAME_HOOKS``, and the variant spec (map data
package, calendar, off-board tiles) in
``entente.engine.variant_meta.VARIANT_SPECS``.
"""
