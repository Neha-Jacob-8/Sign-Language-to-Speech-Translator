import numpy as np
import pytest

from signspeak.features import FEATURE_DIM, augment, featurize, order_slots, pack_hands, to_pixels


def make_hand(rng, x0, y0, size=40.0):
    hand = rng.normal(size=(21, 3)).astype(np.float32) * size / 3
    hand[0] = 0  # wrist at the origin of the hand
    hand[9] = [0, -size, 0]  # middle-finger knuckle above the wrist
    hand[:, 0] += x0
    hand[:, 1] += y0
    return hand


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_feature_dimension(rng):
    slots, present = pack_hands([make_hand(rng, 100, 200)])
    assert featurize(slots[None], present[None]).shape == (1, FEATURE_DIM)


def test_features_ignore_position_and_scale(rng):
    slots, present = pack_hands([make_hand(rng, 100, 200), make_hand(rng, 300, 210)])
    moved = slots * 2.5 + np.array([40, -30, 0], dtype=np.float32)
    np.testing.assert_allclose(featurize(moved[None], present[None]), featurize(slots[None], present[None]), atol=1e-4)


def test_hands_are_ordered_left_to_right(rng):
    left, right = make_hand(rng, 100, 200), make_hand(rng, 300, 200)
    a, _ = pack_hands([left, right])
    b, _ = pack_hands([right, left])
    np.testing.assert_array_equal(a, b)
    assert a[0, 0, 0] < a[1, 0, 0]


def test_single_hand_uses_first_slot(rng):
    slots, present = pack_hands([make_hand(rng, 100, 200)])
    assert present.tolist() == [True, False]
    feats = featurize(slots[None], present[None])[0]
    assert not feats[63:126].any()  # empty second slot
    assert not feats[128:].any()  # no between-hands vector


def test_no_hands_rejected_shape():
    with pytest.raises(ValueError):
        pack_hands([np.zeros((5, 3))])


def test_to_pixels_keeps_aspect_ratio():
    norm = np.array([[0.5, 0.5, 0.1]] * 21, dtype=np.float32)
    px = to_pixels(norm, 640, 480)
    assert px[0].tolist() == pytest.approx([320, 240, 64])


def test_order_slots_moves_lone_hand_to_first_slot(rng):
    hands = np.zeros((1, 2, 21, 3), dtype=np.float32)
    hands[0, 1] = make_hand(rng, 100, 200)
    present = np.array([[False, True]])
    h, p = order_slots(hands, present)
    assert p.tolist() == [[True, False]]
    np.testing.assert_array_equal(h[0, 0], hands[0, 1])


def test_augment_keeps_invariants(rng):
    slots, present = pack_hands([make_hand(rng, 100, 200), make_hand(rng, 300, 210)])
    hands = np.repeat(slots[None], 64, axis=0)
    pres = np.repeat(present[None], 64, axis=0)
    h, p = augment(hands, pres, rng)
    assert h.shape == hands.shape and p.all()
    assert (h[:, 0, 0, 0] <= h[:, 1, 0, 0]).all()  # still left-to-right after mirroring
    assert np.isfinite(featurize(h, p)).all()
