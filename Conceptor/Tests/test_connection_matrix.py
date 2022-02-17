from conceptor_test_class import ConceptorTestBase


class TestConnectionMatrix(ConceptorTestBase):
    def test_shape_loaded_matrix(self) -> None:
        # Asserts that the final connection matrix has shape NxN
        self.assertEqual(self.C.W.shape, (self.C.n, self.C.n))

    def test_shape_initial_matrix(self) -> None:
        # Asserts that the initial connection matrix has shape NxN
        self.assertEqual(self.C.Wstar.shape, (self.C.n, self.C.n))
