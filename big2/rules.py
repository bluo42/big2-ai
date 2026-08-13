"""House-rule variants.

The variant space is deliberately small: each flag is a rule real tables
actually disagree on (see https://www.pagat.com/climbing/bigtwo.html).

- ``allow_triples``:    lone triples are playable as their own size class
                        (compared by rank).  Off in the house rules.
- ``pass_locks``:       passing locks you out for the remainder of the
                        trick.  When off, a pass only skips your turn and
                        you may play again if the trick comes back around.
- ``flush_rank_first``: flushes compare by top ranks before suit (poker
                        style).  Off means the house rule: suit first, so
                        any spade flush beats any heart flush.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleConfig:
    allow_triples: bool = False
    pass_locks: bool = True
    flush_rank_first: bool = False

    def label(self) -> str:
        parts = []
        if self.allow_triples:
            parts.append("triples")
        if not self.pass_locks:
            parts.append("soft-pass")
        if self.flush_rank_first:
            parts.append("flush-by-rank")
        return "+".join(parts) if parts else "house"


DEFAULT_RULES = RuleConfig()
