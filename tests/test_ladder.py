import random
import unittest

from big2.cards import parse_card
from big2.decomposition import DecompositionStrategy, min_plays
from big2.game import Big2Game
from big2.ismcts import ISMCTSStrategy
from big2.rules import RuleConfig
from big2.strategies import FiveCardDumper, PlayLowest, SmartHeuristic


def c(*names):
    return [parse_card(n) for n in names]


class TestDecomposition(unittest.TestCase):
    def test_known_partitions(self):
        # two straights + pair + single -> 4 plays
        hand = c("3d", "4c", "5h", "6s", "7d", "8d", "9c", "10h", "Js", "Qd",
                 "Kd", "Kc", "2s")
        k, part = min_plays(hand)
        self.assertEqual(k, 4)
        self.assertEqual(sorted(x for u in part for x in u.cards), sorted(hand))

        # 13-card single suit: 2 flushes + 3 singles is optimal
        flush_hand = [r * 4 for r in range(13)]
        k, _ = min_plays(flush_hand)
        self.assertEqual(k, 5)

    def test_triples_shrink_decomposition(self):
        hand = c("9d", "9c", "9h")
        k_default, _ = min_plays(hand)
        self.assertEqual(k_default, 2)  # pair + single
        k_triples, part = min_plays(hand, RuleConfig(allow_triples=True))
        self.assertEqual(k_triples, 1)  # one lone triple
        self.assertEqual(len(part), 1)

    def test_partition_is_valid_cover(self):
        rng = random.Random(3)
        from big2.cards import full_deck

        for _ in range(20):
            deck = full_deck()
            rng.shuffle(deck)
            hand = sorted(deck[:13])
            k, part = min_plays(hand)
            covered = sorted(x for u in part for x in u.cards)
            self.assertEqual(covered, hand)
            self.assertEqual(len(part), k)

    def test_strategy_plays_full_games(self):
        for seed in range(5):
            game = Big2Game(rng=random.Random(seed))
            scores = game.play_out(
                [DecompositionStrategy(), SmartHeuristic(), PlayLowest(),
                 FiveCardDumper()]
            )
            self.assertEqual(sum(scores.values()), 0)


class TestISMCTS(unittest.TestCase):
    def test_plays_full_game(self):
        game = Big2Game(rng=random.Random(1))
        scores = game.play_out(
            [ISMCTSStrategy(n_sims=25, seed=1), PlayLowest(), PlayLowest(),
             PlayLowest()]
        )
        self.assertEqual(sum(scores.values()), 0)

    def test_determinization_consistency(self):
        game = Big2Game(rng=random.Random(2))
        strat = ISMCTSStrategy(n_sims=5, seed=2)
        me = game.turn
        world = strat._determinize(game, me)
        self.assertEqual(world.hands[me], game.hands[me])
        for p in range(game.num_players):
            self.assertEqual(len(world.hands[p]), len(game.hands[p]))
        # determinized hands only use cards unseen from me
        seen = set(game.hands[me]) | set(game.played_cards)
        for p in range(game.num_players):
            if p != me:
                self.assertFalse(seen.intersection(world.hands[p]))


class TestDMC(unittest.TestCase):
    def test_smoke_train_and_play(self):
        from big2.dmc import DMCPolicy, train_dmc

        policy = train_dmc(episodes=30, verbose_every=0, seed=0)
        game = Big2Game(rng=random.Random(0))
        scores = game.play_out([policy, PlayLowest(), PlayLowest(), PlayLowest()])
        self.assertEqual(sum(scores.values()), 0)


class TestClone(unittest.TestCase):
    def test_clone_is_independent(self):
        game = Big2Game(rng=random.Random(4))
        clone = game.clone()
        clone.step(clone.legal_moves()[0])
        self.assertNotEqual(
            sum(len(h) for h in clone.hands), sum(len(h) for h in game.hands)
        )
        self.assertEqual(len(game.history), 0)
        self.assertEqual(len(clone.history), 1)


if __name__ == "__main__":
    unittest.main()
