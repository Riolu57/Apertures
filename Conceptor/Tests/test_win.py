from conceptor_test_class import ConceptorTestBase


class TestWin(ConceptorTestBase):
    def test_win_shape(self):
        # Assert shape of Matrix correct
        self.assertEqual(self.C.Win.shape, (self.C.n, 1))
