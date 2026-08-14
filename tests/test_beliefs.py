import random
import unittest

from big2.beliefs import BeliefState, prob_at_least_one
from big2.cards import TWO_RANK, parse_card, rank
from big2.combos import classify
from big2.game import Big2Game
from big2.strategies import PlayLowest


def c(*names):
    return [parse_card(n) for n in names]


class TestAnalyticLayer(unittest.TestCase):
    def test_hypergeometric_helper(self):
        self.assertAlmostEqual(prob_at_least_one(0, 39, 13), 0.0)
        self.assertAlmostEqual(prob_at_least_one(4, 4, 4), 1.0)
        # one target card among 39 unseen, 13-card hand: exactly 1/3
        self.assertAlmostEqual(prob_at_least_one(1, 39, 13), 13 / 39)

    def test_fresh_game_marginals(self):
        game = Big2Game(rng=random.Random(0))
        viewpoint = game.turn
        b = BeliefState(game, viewpoint)
        self.assertEqual(b.n_unseen, 39)
        pmap = b.card_probability_map()
        for p in b.opponents:
            for prob in pmap[p].values():
                self.assertAlmostEqual(prob, 13 / 39)
        # each unseen card is somewhere among the three opponents (4p game)
        for card in b.unseen:
            total = sum(pmap[p][card] for p in b.opponents)
            self.assertAlmostEqual(total, 1.0)

    def test_marginals_sharpen_with_elimination(self):
        game = Big2Game(rng=random.Random(1))
        me = game.turn
        b0 = BeliefState(game, me)
        twos_before = {p: b0.prob_holds_two(p) for p in b0.opponents}
        # play out some tricks, then re-evaluate from the same viewpoint
        policies = [PlayLowest() for _ in range(4)]
        for _ in range(20):
            if game.game_over:
                break
            game.step(policies[game.turn].select(game, game.turn) if game.legal_moves()
                      else None)
        if not game.game_over:
            b1 = BeliefState(game, me)
            self.assertLess(b1.n_unseen, 39)
            played_twos = sum(1 for x in game.played_cards if rank(x) == TWO_RANK)
            mine = sum(1 for x in game.hands[me] if rank(x) == TWO_RANK)
            if played_twos + mine == 4:
                for p in b1.opponents:
                    self.assertEqual(b1.prob_holds_two(p), 0.0)

    def test_known_hand_when_forced(self):
        game = Big2Game(rng=random.Random(2))
        me = game.turn
        # give viewpoint knowledge: artificially move all but one opponent's
        # cards into played (simulating deep endgame)
        opp = (me + 1) % 4
        for p in range(4):
            if p not in (me, opp):
                game.played_cards.extend(game.hands[p])
                game.hands[p] = []
        b = BeliefState(game, me)
        known = b.known_hand(opp)
        self.assertEqual(known, game.hands[opp])
        self.assertIsNone(b.known_hand((me + 2) % 4))


class TestMonteCarloLayer(unittest.TestCase):
    def test_world_consistency(self):
        game = Big2Game(rng=random.Random(3))
        me = game.turn
        b = BeliefState(game, me, rng=random.Random(0))
        for world, weight in b.sample_worlds(20):
            self.assertGreaterEqual(weight, 0.0)
            used = []
            for p in b.opponents:
                self.assertEqual(len(world[p]), len(game.hands[p]))
                used.extend(world[p])
            self.assertEqual(len(used), len(set(used)))
            self.assertTrue(set(used).issubset(set(b.unseen)))

    def test_pass_evidence_downweights(self):
        """A player who passed on a 2s-high single should be believed to
        hold the missing high cards less often than the uniform prior."""
        game = Big2Game(rng=random.Random(4))
        # Fabricate: player 0 viewpoint; player 1 passed on a high single.
        game.hands[0] = c("3d", "4d", "5d")
        game.hands[1] = c("6d", "2h", "2s")  # actually holds beaters
        game.hands[2] = c("7d", "8d", "9d")
        game.hands[3] = c("10d", "Jd", "Qd")
        game.played_cards = [x for x in range(52)
                             if x not in [y for h in game.hands for y in h]]
        table = classify(c("2d"))
        game.history = []
        from big2.game import PlayRecord
        game.history.append(PlayRecord(2, table))
        game.history.append(PlayRecord(1, None))  # player 1 passed on the 2d
        b_honest = BeliefState(game, 0, pass_honesty=0.1, rng=random.Random(0))
        b_uniform = BeliefState(game, 0, pass_honesty=1.0, rng=random.Random(0))
        # P(player 1 holds a card beating 2d) under both models
        beat = lambda p, hand: any(x > parse_card("2d") for x in hand)
        p_honest = b_honest.event_probability(beat, k=400)[1]
        p_uniform = b_uniform.event_probability(beat, k=400)[1]
        self.assertLess(p_honest, p_uniform)

    def test_class_probabilities_shape(self):
        game = Big2Game(rng=random.Random(5))
        b = BeliefState(game, game.turn, rng=random.Random(1))
        probs = b.class_probabilities(k=50)
        for p in b.opponents:
            for key in ("pair", "triple", "bomb", "straight", "flush"):
                self.assertGreaterEqual(probs[p][key], 0.0)
                self.assertLessEqual(probs[p][key], 1.0)
            self.assertGreater(probs[p]["pair"], probs[p]["bomb"])

    def test_prob_can_beat(self):
        game = Big2Game(rng=random.Random(6))
        b = BeliefState(game, game.turn, rng=random.Random(2))
        low = classify(c("3c")) if parse_card("3c") in b.unseen else classify(c("4d"))
        high = classify(c("2s"))
        p_low = b.prob_can_beat(low, k=60)
        p_high = b.prob_can_beat(high, k=60)
        for p in b.opponents:
            self.assertGreaterEqual(p_low[p], p_high[p])
            self.assertEqual(p_high[p], 0.0)  # nothing beats the 2 of spades


if __name__ == "__main__":
    unittest.main()
