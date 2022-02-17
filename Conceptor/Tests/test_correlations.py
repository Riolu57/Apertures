from conceptor_test_class import ConceptorTestBase


class TestCorrelations(ConceptorTestBase):
    def test_shape(self):
        # Assert shape of each R_i is NxN
        for r in self.C.create_r_i():
            self.assertEqual(r.shape, (self.C.n, self.C.n))
