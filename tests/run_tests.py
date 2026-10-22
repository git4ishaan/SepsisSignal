"""
Self-contained test runner using Python standard library unittest.
Zero extra dependencies required.
"""

import os
import tempfile
import unittest
from src.predict import predict
from src.server import parse_and_process_line
from src.store import init_db


class TestAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_sepsis.db")
        init_db(self.db_path)

    def tearDown(self):
        for suffix in ["", "-wal", "-shm"]:
            p = self.db_path + suffix
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def test_check_1_immutability(self):
        """Hour 2 reply is byte-identical before and after Hour 3 arrives."""
        h1 = "H|P002|1|80|120|18|36.5|8.0\n"
        h2 = "H|P002|2|85|118|19|36.8|8.5\n"
        
        reply1 = parse_and_process_line(h1, db_path=self.db_path)
        reply2_before = parse_and_process_line(h2, db_path=self.db_path)
        
        # Send Hour 3 with extreme values
        h3 = "H|P002|3|135|85|32|39.4|18.5\n"
        reply3 = parse_and_process_line(h3, db_path=self.db_path)
        self.assertTrue(reply3.startswith("R|P002|3|"))
        
        # Query Hour 2 again
        reply2_after = parse_and_process_line(h2, db_path=self.db_path)
        self.assertEqual(reply2_before, reply2_after)

    def test_check_2_empty_fields_count(self):
        """H|P001|2|88|112||| returns empty count 3 (> 0) without stripping trailing pipes."""
        line = "H|P001|2|88|112|||\n"
        reply = parse_and_process_line(line, db_path=self.db_path)
        self.assertTrue(reply.startswith("R|P001|2|"))
        tokens = reply.strip().split("|")
        self.assertEqual(len(tokens), 6)
        empty_count = int(tokens[5])
        self.assertEqual(empty_count, 3)

    def test_check_3_streaming_parity(self):
        """Streaming patient through the protocol produces identical results to single predict()."""
        hours = [
            (1, "H|P100|1|78|115|16|36.7|7.2\n"),
            (2, "H|P100|2|82|118|17|36.8|7.5\n"),
            (3, "H|P100|3|90|112|20|37.1|8.9\n"),
        ]
        for hr, line in hours:
            sock_reply = parse_and_process_line(line, db_path=self.db_path)
            tokens = sock_reply.strip().split("|")
            sock_prob = float(tokens[3])
            sock_alert = int(tokens[4])

            pred = predict("P100", hr, db_path=self.db_path)
            self.assertAlmostEqual(pred["probability"], sock_prob, places=4)
            self.assertEqual(pred["alert"], sock_alert)

    def test_check_4_dashboard_clock_parity(self):
        """Clock call to predict() returns same probability and alert as socket reply."""
        line = "H|P004|1|102|95|24|38.2|14.0\n"
        reply = parse_and_process_line(line, db_path=self.db_path)
        tokens = reply.strip().split("|")
        sock_prob = float(tokens[3])
        sock_alert = int(tokens[4])
        
        pred = predict("P004", 1, db_path=self.db_path)
        self.assertEqual(pred["probability"], sock_prob)
        self.assertEqual(pred["alert"], sock_alert)

    def test_check_5_unseen_patient(self):
        """A patient ID never seen before works with no dataset lookup."""
        new_patient_line = "H|UNKNOWN_999|1|92|124|16|37.0|7.5\r\n"
        reply = parse_and_process_line(new_patient_line, db_path=self.db_path)
        self.assertTrue(reply.startswith("R|UNKNOWN_999|1|"))
        tokens = reply.strip().split("|")
        self.assertEqual(len(tokens), 6)
        prob = float(tokens[3])
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_check_6_malformed_lines(self):
        """Malformed lines return E|<reason> and keep running; \\r\\n and \\n accepted."""
        bad_lines = [
            "INVALID_LINE\n",
            "H|P003|1|80|120\n",
            "H|P003|not_an_int|80|120|18|36.5|8.0\n",
            "H|P003|1|abc|120|18|36.5|8.0\r\n",
            "X|P003|1|80|120|18|36.5|8.0\n",
            "\n",
        ]
        for bad in bad_lines:
            reply = parse_and_process_line(bad, db_path=self.db_path)
            self.assertTrue(reply.startswith("E|"), f"Failed for: {bad}")

    def test_out_of_order_causality(self):
        """Causal filtering: Hour 5 observation does not leak into Hour 2."""
        parse_and_process_line("H|P005|1|80|120|18|36.5|8.0\n", db_path=self.db_path)
        parse_and_process_line("H|P005|5|130|80|30|39.5|20.0\n", db_path=self.db_path)
        
        # Hour 2 arrives out of order
        reply_h2 = parse_and_process_line("H|P005|2|82|119|18|36.6|8.1\n", db_path=self.db_path)
        self.assertTrue(reply_h2.startswith("R|P005|2|"))
        
        p2 = predict("P005", 2, db_path=self.db_path)
        self.assertTrue(p2["probability"] < 0.8)


def load_tests(loader, standard_tests, pattern):
    """
    Standard unittest discovery hook.
    Builds the canonical full acceptance and regression suite:
    - TestAcceptance (7 jury and causal checks)
    - TestFutureRowInvariance (9 future-row invariance checks)
    - TestNonFiniteHandling (5 non-finite measurement regression checks)
    Total: 21 tests.
    """
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAcceptance))
    from tests.test_t02_future_row_invariance import TestFutureRowInvariance
    suite.addTests(loader.loadTestsFromTestCase(TestFutureRowInvariance))
    from tests.test_non_finite_regression import TestNonFiniteHandling
    suite.addTests(loader.loadTestsFromTestCase(TestNonFiniteHandling))
    from tests.test_t05_prediction_immutability import TestT05PredictionImmutability
    suite.addTests(loader.loadTestsFromTestCase(TestT05PredictionImmutability))
    from tests.test_t06_socket_server import TestT06SocketProtocolAndResilience
    suite.addTests(loader.loadTestsFromTestCase(TestT06SocketProtocolAndResilience))
    from tests.test_t07_model_integrity import TestT07ModelLoadingAndIntegrity
    suite.addTests(loader.loadTestsFromTestCase(TestT07ModelLoadingAndIntegrity))
    from tests.test_t08_metric_reproduction import TestT08MetricReproduction
    suite.addTests(loader.loadTestsFromTestCase(TestT08MetricReproduction))
    from tests.test_t09_temporal_robustness import TestT09TemporalRobustness
    suite.addTests(loader.loadTestsFromTestCase(TestT09TemporalRobustness))
    from tests.test_ensemble import TestSepsisEnsemble
    suite.addTests(loader.loadTestsFromTestCase(TestSepsisEnsemble))
    from tests.test_ensemble_audit import TestEnsembleAudit
    suite.addTests(loader.loadTestsFromTestCase(TestEnsembleAudit))
    from tests.test_benchmark_utilities import TestBenchmarkUtilities
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarkUtilities))
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
