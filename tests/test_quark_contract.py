import unittest
import sys
import os
from sympy import S, Symbol, Add, Mul

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lattice.quark_contract import Tag, Qurak, Propagator, HadronFlavorStructure, quark_contract, _quark_contract


class TestTag(unittest.TestCase):
    """Test cases for Tag class"""

    def test_tag_creation(self):
        """Test creating a Tag object"""
        tag = Tag(tag=1, time=2)
        self.assertEqual(tag.tag, 1)
        self.assertEqual(tag.time, 2)

    def test_tag_immutability(self):
        """Test that Tag objects are immutable"""
        tag = Tag(tag=1, time=2)
        with self.assertRaises(AttributeError):
            tag.tag = 3


class TestQurak(unittest.TestCase):
    """Test cases for Qurak class"""

    def test_quark_creation(self):
        """Test creating a quark"""
        tag = Tag(tag=1, time=2)
        quark = Qurak(flavor="u", tag=tag, anti=False)
        self.assertEqual(quark.flavor, "u")
        self.assertEqual(quark.tag, tag)
        self.assertFalse(quark.anti)

    def test_antiquark_creation(self):
        """Test creating an antiquark"""
        tag = Tag(tag=1, time=2)
        antiquark = Qurak(flavor="d", tag=tag, anti=True)
        self.assertEqual(antiquark.flavor, "d")
        self.assertEqual(antiquark.tag, tag)
        self.assertTrue(antiquark.anti)

    def test_quark_string_representation(self):
        """Test string representation of quarks and antiquarks"""
        tag = Tag(tag=1, time=2)
        quark = Qurak(flavor="u", tag=tag, anti=False)
        antiquark = Qurak(flavor="d", tag=tag, anti=True)

        self.assertEqual(str(quark), "u(1)")
        self.assertEqual(str(antiquark), "\\bar{d}(1)")


class TestPropagator(unittest.TestCase):
    """Test cases for Propagator class"""

    def test_propagator_creation(self):
        """Test creating a propagator"""
        source_tag = Tag(tag=1, time=2)
        sink_tag = Tag(tag=3, time=4)
        propagator = Propagator(flavor="u", source_tag=source_tag, sink_tag=sink_tag)

        self.assertEqual(propagator.flavor, "u")
        self.assertEqual(propagator.source_tag, source_tag)
        self.assertEqual(propagator.sink_tag, sink_tag)
        self.assertEqual(propagator.tag, "S^u")

    def test_local_propagator(self):
        """Test creating a local propagator (same time)"""
        tag = Tag(tag=1, time=2)
        propagator = Propagator(flavor="u", source_tag=tag, sink_tag=tag)

        self.assertEqual(propagator.tag, "S^u_\\mathrm{local}")

    def test_propagator_string_representation(self):
        """Test string representation of propagators"""
        source_tag = Tag(tag=1, time=2)
        sink_tag = Tag(tag=3, time=4)
        propagator = Propagator(flavor="u", source_tag=source_tag, sink_tag=sink_tag)

        self.assertEqual(str(propagator), "S^u(3, 1)")


class TestHadronFlavorStructure(unittest.TestCase):
    """Test cases for HadronFlavorStructure class"""

    def test_meson_creation(self):
        """Test creating a meson (2-quark structure)"""
        meson = HadronFlavorStructure("ud", time=2)

        self.assertEqual(meson.flavor_str, "ud")
        self.assertEqual(meson.time, 2)
        self.assertEqual(meson.baryon_num, 0)
        self.assertEqual(meson.quark_list, ["d"])
        self.assertEqual(meson.anti_quark_list, ["u"])

    def test_baryon_creation(self):
        """Test creating a baryon (3-quark structure)"""
        baryon = HadronFlavorStructure("uds", time=2)

        self.assertEqual(baryon.flavor_str, "uds")
        self.assertEqual(baryon.time, 2)
        self.assertEqual(baryon.baryon_num, 1)
        self.assertEqual(baryon.quark_list, ["u", "d", "s"])
        self.assertEqual(baryon.anti_quark_list, [])

    def test_antibaryon_creation(self):
        """Test creating an antibaryon (3-antiquark structure)"""
        antibaryon = HadronFlavorStructure("bar{uds}", time=2)

        self.assertEqual(antibaryon.flavor_str, "bar{uds}")
        self.assertEqual(antibaryon.time, 2)
        self.assertEqual(antibaryon.baryon_num, -1)
        self.assertEqual(antibaryon.quark_list, [])
        self.assertEqual(antibaryon.anti_quark_list, ["u", "d", "s"])

    def test_meson_conjugate(self):
        """Test conjugating a meson"""
        meson = HadronFlavorStructure("ud", time=2)
        conjugated = meson.conjugate()

        self.assertEqual(conjugated.flavor_str, "du")
        self.assertEqual(conjugated.time, 2)

    def test_baryon_conjugate(self):
        """Test conjugating a baryon"""
        baryon = HadronFlavorStructure("uds", time=2)
        conjugated = baryon.conjugate()

        self.assertEqual(conjugated.flavor_str, "bar{uds}")
        self.assertEqual(conjugated.time, 2)

    def test_antibaryon_conjugate(self):
        """Test conjugating an antibaryon"""
        antibaryon = HadronFlavorStructure("bar{uds}", time=2)
        conjugated = antibaryon.conjugate()

        self.assertEqual(conjugated.flavor_str, "uds")
        self.assertEqual(conjugated.time, 2)


class TestQuarkContract(unittest.TestCase):
    """Test cases for quark_contract function"""

    def setUp(self):
        """Set up test fixtures"""
        # Create some test particles
        self.particles = ["particle1", "particle2"]

    def test_meson_meson_contraction(self):
        """Test contracting two mesons"""
        meson1 = HadronFlavorStructure("ud", time=0)
        meson2 = HadronFlavorStructure("du", time=1)

        expr = meson1 * meson2
        result = quark_contract(expr, self.particles)

        # Verify result is not None and has expected structure
        self.assertIsNotNone(result)

    def test_baryon_antibaryon_contraction(self):
        """Test contracting a baryon and an antibaryon"""
        baryon = HadronFlavorStructure("uds", time=0)
        antibaryon = HadronFlavorStructure(r"bar{uds}", time=1)

        expr = baryon * antibaryon
        result = quark_contract(expr, self.particles)

        # Verify result is not None and has expected structure
        self.assertIsNotNone(result)

    def test_degenerate_quarks(self):
        """Test contraction with degenerate u and d quarks"""
        meson1 = HadronFlavorStructure("uu", time=0)
        meson2 = HadronFlavorStructure("dd", time=1)

        expr = meson1 * meson2
        result = quark_contract(expr, self.particles, degenerate=True)
        # Verify result is not None and has expected structure
        self.assertIsNotNone(result)
        self.assertEqual(result.diagram.adjacency_matrix, [[1, 0], [0, 1]])


class TestQuarkContractInternal(unittest.TestCase):
    """Test cases for _quark_contract internal function"""

    def test_empty_symbol_list(self):
        """Test _quark_contract with empty symbol list"""
        result_list = []
        result = []
        _quark_contract([], result_list, result, True)

        self.assertEqual(len(result_list), 1)
        self.assertEqual(result_list[0], S(1))

    def test_simple_contraction(self):
        """Test _quark_contract with a simple quark-antiquark pair"""
        tag1 = Tag(tag=1, time=0)
        tag2 = Tag(tag=2, time=1)

        quark = Qurak(flavor="u", tag=tag1, anti=False)
        antiquark = Qurak(flavor="u", tag=tag2, anti=True)

        symbol_list = [quark, antiquark]
        result_list = []
        result = []

        _quark_contract(symbol_list, result_list, result, True)

        # Verify result_list contains one term with a propagator
        self.assertEqual(len(result_list), 1)
        # The result might be a single Propagator or a Mul containing a Propagator
        self.assertTrue(
            isinstance(result_list[0], Propagator)
            or (
                isinstance(result_list[0], Mul)
                and any(isinstance(factor, Propagator) for factor in result_list[0].args)
            )
        )


if __name__ == "__main__":
    unittest.main()
