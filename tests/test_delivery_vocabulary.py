"""Guards on the delivery vocabulary itself (`deliveries/vocabulary.py`).

The module's own docstring sets its boundary: *"What is enforced here is only what a single
value can be wrong about on its own."* Cross-file rules — does the reconciliation graph
connect, do the targets add up — live in `deliveries/coherence.py` and are tested in
`tests/test_delivery_coherence.py`.

This file exists because `provides` (#427) is the first field on `Source` since the module
was written, and the thing most likely to go wrong with it is silent: a value that is
accepted but stored in a shape nothing downstream expects. `send` already normalises a list
to a tuple in `Delivery.__post_init__`; `provides` has to do the same, on a different class,
and nothing was asserting either.
"""

import pytest

from deliveries.vocabulary import Source, cm, pgm

pytestmark = [pytest.mark.green]


class TestProvidesIsOptionalAndInert:
    """`None` means "every target this source contains", so nothing existing changes."""

    def test_omitting_it_gives_none(self):
        assert pgm("rusty_bucket").provides is None
        assert cm("pink_ponyclub").provides is None

    def test_the_one_source_form_is_unchanged(self):
        """Both real deliveries are `send=[pgm("rusty_bucket")]`. If this breaks, they do."""
        assert pgm("rusty_bucket") == Source(name="rusty_bucket", level="pgm")

    def test_both_level_wrappers_carry_it(self):
        assert pgm("x", provides=("a",)).provides == ("a",)
        assert cm("y", provides=("b",)).provides == ("b",)


class TestTheShapeItIsStoredIn:
    """A list that stays a list is the silent failure: it compares unequal to a tuple,
    and an unhashable field breaks a frozen dataclass's `__hash__`."""

    def test_a_list_is_normalised_to_a_tuple(self):
        assert pgm("x", provides=["a", "b"]).provides == ("a", "b")
        assert isinstance(pgm("x", provides=["a", "b"]).provides, tuple)

    def test_normalisation_happens_on_the_class_not_only_the_helpers(self):
        """`Source(...)` constructed directly must behave as `pgm(...)` does — the
        helpers are a convenience, not the enforcement point."""
        assert Source(name="x", level="pgm", provides=["a"]).provides == ("a",)

    def test_a_source_with_provides_is_still_hashable(self):
        """`Source` is `frozen=True`; a list field would make it unhashable and the
        failure would surface far away, in whatever first puts one in a set."""
        assert len({pgm("x", provides=["a"]), pgm("x", provides=["a"])}) == 1

    def test_it_is_still_frozen(self):
        with pytest.raises(Exception) as caught:
            pgm("x").name = "other"
        assert "Frozen" in type(caught.value).__name__


class TestTheOneSlipItRefuses:
    def test_a_bare_string_is_refused_not_split_into_characters(self):
        """`str` is a `Sequence[str]`, so `provides="lr_ged_sb"` would normalise to
        ('l','r','_','g',...) and later refuse the delivery for reasons no one could
        read. Writing one target without the trailing comma is an easy slip.

        `Delivery.__post_init__` already catches the same mistake on `send`
        ("send must be a list, even with one source"); this is that guard one field over.
        """
        with pytest.raises(TypeError) as caught:
            pgm("rusty_bucket", provides="lr_ged_sb")
        message = str(caught.value)
        assert "Write:" in message, "the module's raises must say what to write"
        assert "lr_ged_sb" in message
        assert "rusty_bucket" in message, "name the source, not just the value"

    def test_a_tuple_of_one_is_accepted(self):
        """The corrected form from that error message must actually work."""
        assert pgm("rusty_bucket", provides=("lr_ged_sb",)).provides == ("lr_ged_sb",)


class TestItIsAClaimAndNotACheck:
    """ADR-019 §3: `pgm("x")` states what you believe; the system refuses if the source
    disagrees. `provides` is the same kind of claim one axis over, and this module
    deliberately verifies neither."""

    def test_an_unknown_target_name_is_accepted_here(self):
        """Whether a target is real needs a run's manifests (register C-123:
        `rusty_bucket` declares `lr_*_best` while both deliveries require `lr_ged_*`).
        Refusing here would refuse correct delivery files."""
        assert pgm("rusty_bucket", provides=("not_a_real_target",)).provides == (
            "not_a_real_target",
        )

    def test_an_unknown_source_name_is_accepted_here(self):
        """Same boundary, already true before `provides` — pinned so a later change
        cannot quietly move source resolution into this module."""
        assert pgm("no_such_ensemble").name == "no_such_ensemble"

    def test_an_empty_provides_is_not_the_same_as_omitting_it(self):
        """`()` claims nothing; `None` claims everything. Collapsing them would make
        "this source provides no targets" unsayable, and #428's coverage rule needs to
        tell the two apart."""
        assert pgm("x", provides=()).provides == ()
        assert pgm("x").provides is None

    def test_an_empty_LIST_still_normalises(self):
        """The guard must be `is not None`, not truthiness.

        Found by mutation: changing `if self.provides is not None:` to `if self.provides:`
        passed every other test here, because an empty *tuple* normalises to itself either
        way. An empty *list* does not — it stays a list, which is unhashable and compares
        unequal to `()`. `provides=[]` is a plausible thing to write while editing a
        delivery file down to one source.
        """
        empty = pgm("x", provides=[])
        assert empty.provides == ()
        assert isinstance(empty.provides, tuple)
        assert len({empty, pgm("x", provides=[])}) == 1  # unhashable if it stayed a list
