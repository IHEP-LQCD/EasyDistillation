import unittest
import sys
import os
import numpy as np
from sympy import Add, Mul, Symbol

# Add project root directory to Python path
import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from lattice.quark_diagram import QuarkDiagram, Diagram, diagram_simplify


# Mock Propagator and Vertex classes for testing
class MockPropagator:
    def __init__(self, name):
        self.name = name

    def get(self, t_source, t_sink):
        return np.ones((4, 4))  # Return a simple matrix

    def __str__(self):
        return f"MockPropagator({self.name})"

    def __repr__(self):
        return f"MockPropagator({self.name})"

    def __eq__(self, other):
        if not isinstance(other, MockPropagator):
            return False
        return self.name == other.name

    def __lt__(self, other):
        if not isinstance(other, MockPropagator):
            return NotImplemented
        return self.name < other.name

    def __hash__(self):
        return hash(self.name)


class MockVertex:
    def __init__(self, name):
        self.name = name

    def get(self, t):
        return np.ones((4, 4))  # Return a simple matrix

    def __str__(self):
        return f"MockVertex({self.name})"

    def __repr__(self):
        return f"MockVertex({self.name})"

    def __eq__(self, other):
        if not isinstance(other, MockVertex):
            return False
        return self.name == other.name

    def __lt__(self, other):
        if not isinstance(other, MockVertex):
            return NotImplemented
        return self.name < other.name

    def __hash__(self):
        return hash(self.name)


class TestSimplify(unittest.TestCase):

    def test_already_simplified_diagram(self):
        """Test already simplified Diagram object"""
        # Create an already simplified Diagram object
        diagram = Diagram(
            QuarkDiagram([[0, 1], [0, 0]]),
            [0, 1],
            [MockVertex("v1"), MockVertex("v2")],
            [None, MockPropagator("prop1")],
        )
        result = diagram_simplify(diagram)
        self.assertEqual(result, diagram)

    def test_sort_vertex(self):
        """Test vertex sorting"""
        # Create a Diagram object for vertex sorting
        diagram = Diagram(
            QuarkDiagram([[0, 1], [0, 0]]),
            [1, 0],
            [MockVertex("v2"), MockVertex("v1")],
            [None, MockPropagator("prop1")],
        )
        expected_result = Diagram(
            QuarkDiagram([[0, 0], [1, 0]]),
            [0, 1],
            [MockVertex("v1"), MockVertex("v2")],
            [None, MockPropagator("prop1")],
        )
        result = diagram_simplify(diagram)
        self.assertEqual(result, expected_result)

    def test_sort_vertex_with_same_vertex(self):
        """Test vertex sorting"""
        # Create a Diagram object for vertex sorting
        diagram = Diagram(
            QuarkDiagram([[0, 0, 1], [0, [[1, 1, 1]], 0], [0, 1, 0]]),
            [0, 0, 0],
            [MockVertex("v1"), MockVertex("v2"), MockVertex("v2")],
            [None, MockPropagator("prop1")],
        )
        expected_result = Diagram(
            QuarkDiagram([[0, 1, 0], [0, 0, 1], [0, 0, [[1, 1, 1]]]]),
            [0, 0, 0],
            [MockVertex("v1"), MockVertex("v2"), MockVertex("v2")],
            [None, MockPropagator("prop1")],
        )
        result = diagram_simplify(diagram)
        self.assertEqual(result, expected_result)

    def test_remove_redundant_vertex(self):
        """Test removing redundant vertices"""
        # Create a diagram with redundant vertices
        adjacency_matrix = [[0, 0, 1], [0, 0, 0], [0, 0, 0]]
        quark_diagram = QuarkDiagram(adjacency_matrix)

        # Create mock objects
        propagator1 = MockPropagator("prop1")
        vertex1 = MockVertex("v1")
        vertex2 = MockVertex("v2")
        redundant_vertex = MockVertex("redundant")

        # Create Diagram instance with redundant vertex
        diagram = Diagram(
            quark_diagram,
            [0, 1, 2],  # time_list
            [vertex1, redundant_vertex, vertex2],  # vertex_list
            [None, propagator1],  # propagator_list
        )

        # Expected result - diagram after removing redundant vertex
        expected_diagram = Diagram(
            QuarkDiagram([[0, 1], [0, 0]]),
            [0, 2],  # time_list
            [vertex1, vertex2],  # vertex_list
            [None, propagator1],  # propagator_list
        )

        # Call simplify function
        result = diagram_simplify(diagram)

        # Verify redundant vertex is removed
        self.assertEqual(result, expected_diagram)

    def test_already_simplified_diagram_with_baryon_vertex(self):
        """Test already simplified Diagram object"""
        # Create an already simplified Diagram object
        diagram = Diagram(
            QuarkDiagram([[0, 1], [1, 0]]),
            [0, 0],
            [MockVertex("v1"), MockVertex("v2")],
            [None, MockPropagator("prop1")],
        )
        result = diagram_simplify(diagram)
        self.assertEqual(result, diagram)

    def test_split_components(self):
        """Test splitting connected components"""
        # Create a diagram with two connected components
        adjacency_matrix = [
            [0, 1, 0, 0],  # Vertex 0 connects to vertex 1
            [1, 0, 0, 0],
            [0, 0, 0, 0],  # Vertex 4 connects to itself
            [0, 0, 0, 2],
        ]
        quark_diagram = QuarkDiagram(adjacency_matrix)

        # Create mock objects
        propagator1 = MockPropagator("prop1")
        propagator2 = MockPropagator("prop2")
        vertex1 = MockVertex("v1")
        vertex2 = MockVertex("v2")
        vertex3 = MockVertex("v3")
        vertex4 = MockVertex("v4")

        diagram = Diagram(
            quark_diagram,
            [0, 1, 2, 3],  # time_list
            [vertex1, vertex2, vertex3, vertex4],  # vertex_list
            [None, propagator1, propagator2],  # propagator_list
        )
        expected_result = Diagram(
            QuarkDiagram([[0, 1], [1, 0]]),
            [0, 1],  # time_list
            [vertex1, vertex2],  # vertex_list
            [None, propagator1],  # propagator_list
        ) * Diagram(
            QuarkDiagram([[1]]),
            [3],  # time_list
            [vertex4],  # vertex_list
            [None, propagator2],  # propagator_list
        )
        result = diagram_simplify(diagram)
        self.assertEqual(result, expected_result)

    def test_complex_expression(self):
        """Test complex expression simplification"""
        """Test complex expression simplification"""
        # Use diagram and objects from previous tests

        # Get diagram and objects from test_split_components
        adjacency_matrix1 = [
            [0, 1, 0, 0],  # Vertex 0 connects to vertex 1
            [1, 0, 0, 0],
            [0, 0, 0, 0],  # Vertex 2 has no connections
            [0, 0, 0, 2],  # Vertex 3 connects to itself
        ]
        quark_diagram1 = QuarkDiagram(adjacency_matrix1)

        propagator1 = MockPropagator("prop1")
        propagator2 = MockPropagator("prop2")
        vertex1 = MockVertex("v1")
        vertex2 = MockVertex("v2")
        vertex3 = MockVertex("v3")
        vertex4 = MockVertex("v4")

        diagram1 = Diagram(
            quark_diagram1,
            [0, 1, 2, 3],  # time_list
            [vertex1, vertex2, vertex3, vertex4],  # vertex_list
            [None, propagator1, propagator2],  # propagator_list
        )

        # Get diagram and objects from test_remove_redundant
        adjacency_matrix2 = [
            [0, 1, 0],  # Vertex 0 connects to vertex 1
            [1, 0, 0],
            [0, 0, 0],  # Vertex 2 is redundant
        ]
        quark_diagram2 = QuarkDiagram(adjacency_matrix2)

        diagram2 = Diagram(
            quark_diagram2,
            [0, 1, 2],  # time_list
            [vertex1, vertex2, vertex3],  # vertex_list
            [None, propagator1],  # propagator_list
        )

        # Create complex expression: diagram1 + 2 * diagram2 + diagram1 * diagram2
        complex_expr = diagram1 + 2 * diagram2 + diagram1 * diagram2

        # Expected result: based on expected_result from previous tests
        expected_diagram1_simplified = Diagram(
            QuarkDiagram([[0, 1], [1, 0]]),
            [0, 1],  # time_list
            [vertex1, vertex2],  # vertex_list
            [None, propagator1],  # propagator_list
        ) * Diagram(
            QuarkDiagram([[1]]),
            [3],  # time_list
            [vertex4],  # vertex_list
            [None, propagator2],  # propagator_list
        )

        expected_diagram2_simplified = Diagram(
            QuarkDiagram([[0, 1], [1, 0]]),
            [0, 1],  # time_list
            [vertex1, vertex2],  # vertex_list
            [None, propagator1],  # propagator_list
        )

        # Expected result: expected_diagram1_simplified + 2 * expected_diagram2_simplified + expected_diagram1_simplified * expected_diagram2_simplified
        expected_result = (
            expected_diagram1_simplified
            + 2 * expected_diagram2_simplified
            + expected_diagram1_simplified * expected_diagram2_simplified
        )

        # Execute test
        result = diagram_simplify(complex_expr)
        self.assertEqual(result, expected_result)

    def test_complex_data_structure(self):
        """Test complex data structure simplification"""
        # Use various test cases from previous tests

        # Test case 1: Simple diagram (test_already_simplified_diagram)
        simple_diagram = Diagram(
            QuarkDiagram([[0, 1], [0, 0]]),
            [0, 1],
            [MockVertex("v1"), MockVertex("v2")],
            [None, MockPropagator("prop1")],
        )

        # Test case 2: Vertex sorting (test_sort_vertex)
        unsorted_diagram = Diagram(
            QuarkDiagram([[0, 1], [0, 0]]),
            [1, 0],
            [MockVertex("v2"), MockVertex("v1")],
            [None, MockPropagator("prop1")],
        )

        # Test case 3: Diagram with same vertices (test_sort_vertex_with_same_vertex)
        same_vertex_diagram = Diagram(
            QuarkDiagram([[0, 1, 0], [0, 0, 1], [0, 0, [[1, 1, 1]]]]),
            [0, 0, 0],
            [MockVertex("v1"), MockVertex("v2"), MockVertex("v2")],
            [None, MockPropagator("prop1")],
        )

        # Test case 4: Diagram with redundant vertices (test_remove_redundant_vertex)
        redundant_diagram = Diagram(
            QuarkDiagram([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
            [0, 1, 2],
            [MockVertex("v1"), MockVertex("redundant"), MockVertex("v2")],
            [None, MockPropagator("prop1")],
        )

        # Test case 5: Diagram with baryon vertex (test_already_simplified_diagram_with_baryon_vertex)
        baryon_diagram = Diagram(
            QuarkDiagram([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0], [0], [1]]]]),
            [0, 0],
            [MockVertex("v1"), MockVertex("v2")],
            [None, MockPropagator("prop1")],
        )

        # Test case 6: Diagram with two connected components (test_split_components)
        split_diagram = Diagram(
            QuarkDiagram(
                [
                    [0, 1, 0, 0],  # Vertex 0 connects to vertex 1
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],  # Vertex 2 does not connect
                    [0, 0, 0, 2],  # Vertex 3 connects to itself
                ]
            ),
            [0, 1, 2, 3],
            [MockVertex("v1"), MockVertex("v2"), MockVertex("v3"), MockVertex("v4")],
            [None, MockPropagator("prop1"), MockPropagator("prop2")],
        )

        # Test case 7: Complex expression (test_complex_expression)
        complex_expr = split_diagram + 2 * redundant_diagram

        # Build complex data structure
        complex_list = [
            simple_diagram,  # Single simple diagram
            [unsorted_diagram, same_vertex_diagram],  # Nested list
            {"redundant": redundant_diagram, "baryon": baryon_diagram},  # Dictionary
            (split_diagram, (complex_expr,)),  # Nested tuple
            np.array([[simple_diagram, unsorted_diagram], [same_vertex_diagram, redundant_diagram]]),  # NumPy array
            simple_diagram * unsorted_diagram + same_vertex_diagram * 3,  # Complex expression
        ]

        # Execute test
        result = diagram_simplify(complex_list)

        # Verify result
        # Only verify list structure and type is preserved, not specific content
        self.assertEqual(len(result), len(complex_list))
        self.assertTrue(isinstance(result[0], Diagram))
        self.assertTrue(isinstance(result[1], list))
        self.assertTrue(isinstance(result[2], dict))
        self.assertTrue(isinstance(result[3], tuple))
        self.assertTrue(isinstance(result[4], np.ndarray))
        self.assertTrue(isinstance(result[5], Add))


if __name__ == "__main__":
    unittest.main()
    # Run single test case
    # test_suite = unittest.TestSuite()
    # test_suite.addTest(TestSimplify("test_complex_data_structure"))
    # unittest.TextTestRunner().run(test_suite)
