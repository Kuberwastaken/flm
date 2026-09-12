"""Small execution-amendment checks, with no fitting or corpus access."""
import copy
import unittest
from scripts import selection_mac_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def test_only_native_initialization_fields_may_change(self):
        parent = dict(graph={'edges': 17}, training_seed=42, initial_state_sha256='old',
                      initial_parameters_sha256='old', initial_nonedge_parameters_sha256='old')
        native = dict(parent, initial_state_sha256='new', initial_parameters_sha256='new',
                      initial_nonedge_parameters_sha256='new')
        runtime.validate_condition(parent, native)
        for key, value in [('graph', {'edges': 18}), ('training_seed', 43)]:
            changed = copy.deepcopy(native)
            changed[key] = value
            with self.assertRaises(ValueError):
                runtime.validate_condition(parent, changed)
        with self.assertRaises(ValueError):
            runtime.validate_condition(parent, dict(native, extra='unregistered'))

    def test_adapter_retains_original_computation_functions(self):
        coordinator, evaluator = runtime.coordinator, runtime.evaluator
        before = (coordinator.fit_case, coordinator.train_study, coordinator.freeze_selection,
                  evaluator.restore_selected, evaluator.score_study, evaluator.summarize)
        runtime.install_adapter()
        after = (coordinator.fit_case, coordinator.train_study, coordinator.freeze_selection,
                 evaluator.restore_selected, evaluator.score_study, evaluator.summarize)
        self.assertEqual(before, after)
        self.assertEqual(coordinator.STUDY, runtime.STUDY)
        self.assertEqual(evaluator.SELECTION, runtime.REPORTS/'study-selection.json')
        self.assertIs(evaluator.verified_context, runtime.verified_context)


if __name__ == '__main__':
    unittest.main()
