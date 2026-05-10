import math
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import vinecopula as vc


class FamilyTests(unittest.TestCase):
    def test_family_names_round_trip(self):
        self.assertEqual(vc.BiCopName(3), "C")
        self.assertEqual(vc.BiCopName("Clayton"), 3)
        np.testing.assert_array_equal(vc.BiCopName(["Gaussian", "t"]), np.array([1, 2]))

    def test_named_familysets(self):
        self.assertEqual(vc.resolve_familyset("recommended"), list(vc.FAST3_FAMILIES))
        self.assertEqual(vc.resolve_familyset("fast3"), [0, 1, 5])
        self.assertIn(2, vc.resolve_familyset("all"))

    def test_tau_round_trip_one_parameter(self):
        for family, tau in [(1, 0.4), (3, 0.4), (4, 0.4), (5, 0.4), (6, 0.4), (23, -0.3)]:
            par = vc.BiCopTau2Par(family, tau, check_taus=False)
            got = vc.BiCopPar2Tau(family, par, check_pars=False)
            self.assertAlmostEqual(float(got), tau, places=6)


class BiCopTests(unittest.TestCase):
    def test_clayton_h_inverse(self):
        cop = vc.BiCop(3, 2.0)
        y = vc.BiCopHinv1(0.2, 0.7, cop)
        self.assertAlmostEqual(vc.BiCopHfunc1(0.2, y, cop), 0.7, places=7)
        x = vc.BiCopHinv2(0.3, 0.8, cop)
        self.assertAlmostEqual(vc.BiCopHfunc2(x, 0.8, cop), 0.3, places=7)

    def test_pdf_and_cdf_shapes(self):
        cop = vc.BiCop(4, 1.5)
        u = np.array([0.2, 0.4, 0.8])
        v = np.array([0.3, 0.5, 0.7])
        self.assertEqual(vc.BiCopPDF(u, v, cop).shape, (3,))
        self.assertEqual(vc.BiCopCDF(u, v, cop).shape, (3,))
        self.assertGreater(vc.BiCopPDF(0.5, 0.5, cop), 0)

    def test_simulation(self):
        sim = vc.BiCopSim(10, 3, 2.0, random_state=42)
        self.assertEqual(sim.shape, (10, 2))
        self.assertTrue(np.all((sim > 0) & (sim < 1)))

    def test_select_accepts_named_familyset(self):
        u = vc.BiCopSim(60, 1, 0.5, random_state=8)
        fit = vc.BiCopSelect(u[:, 0], u[:, 1], familyset="recommended")
        self.assertIn(fit.family, vc.FAST3_FAMILIES)


class StatsTests(unittest.TestCase):
    def test_pobs(self):
        x = np.array([[3, 1], [1, 2], [2, 3]], dtype=float)
        got = vc.pobs(x)
        expected = np.array([[0.75, 0.25], [0.25, 0.5], [0.5, 0.75]])
        np.testing.assert_allclose(got, expected)

    def test_emp_cdf_bounds(self):
        fn = vc.EmpCDF([1, 2, 3])
        self.assertAlmostEqual(fn(-99), 0.25)
        self.assertAlmostEqual(fn(99), 0.75)


class RVineTests(unittest.TestCase):
    def test_two_dimensional_rvine_matches_bicop(self):
        matrix = np.array([[2, 0], [1, 1]])
        family = np.array([[0, 0], [3, 0]])
        par = np.array([[0.0, 0.0], [2.0, 0.0]])
        rvm = vc.RVineMatrix(matrix, family, par)
        u = vc.BiCopSim(25, 3, 2.0, random_state=1)
        self.assertAlmostEqual(
            vc.RVineLogLik(u, rvm)["loglik"],
            vc.BiCopLogLik(u[:, 0], u[:, 1], 3, 2.0),
            places=8,
        )

    def test_two_dimensional_rvine_sim(self):
        matrix = np.array([[2, 0], [1, 1]])
        family = np.array([[0, 0], [3, 0]])
        par = np.array([[0.0, 0.0], [2.0, 0.0]])
        rvm = vc.RVineMatrix(matrix, family, par)
        sim = vc.RVineSim(8, rvm, random_state=3)
        self.assertEqual(sim.shape, (8, 2))
        self.assertTrue(np.all((sim > 0) & (sim < 1)))

    def test_dissmann_structure_select(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(60, 5))
        model = vc.RVineStructureSelect(x, trunc_lvl=3, familyset=[0, 1, 3, 4, 5, 6])
        self.assertIsInstance(model, vc.DissmannVine)
        self.assertEqual([len(tree) for tree in model.edges], [4, 3, 2])
        self.assertTrue(np.isfinite(model.loglik(vc.pobs(x))["loglik"]))

    def test_dissmann_named_familyset_and_tree_count(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(70, 5))
        model = vc.RVineStructureSelect(x, n_trees=2, familyset="fast3")
        self.assertEqual(model.n_layers, 2)
        self.assertEqual(model.n_trees, 2)
        self.assertEqual(model.trunc_lvl, 2)
        self.assertEqual(model.familyset, [0, 1, 5])
        self.assertEqual([len(tree) for tree in model.edges], [4, 3])

    def test_load_fred_md_csv(self):
        content = (
            "sasdate,A,B\n"
            "Transform:,5,2\n"
            "1/1/2000,10,2\n"
            "2/1/2000,11,4\n"
            "3/1/2000,13,7\n"
            "4/1/2000,16,11\n"
        )
        path = pathlib.Path(__file__).resolve().parents[1] / "tests" / "_tmp_fred.csv"
        try:
            path.write_text(content, encoding="utf-8")
            fred = vc.load_fred_md_csv(path, apply_transforms=True, max_missing=0.5)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(fred.data.shape, (3, 2))
        self.assertEqual(fred.names, ["A", "B"])
        self.assertEqual(fred.transform_codes, {"A": 5, "B": 2})


if __name__ == "__main__":
    unittest.main()
