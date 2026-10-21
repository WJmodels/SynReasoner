"""Run with unittest; checks use chemically equivalent and invalid predictions."""
import unittest
import importlib.util
from pathlib import Path
import sys

def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

metrics = pipeline = voting = load_script("synreasoner_test_script", "test.py")

class ChemistryChecks(unittest.TestCase):
    def test_answer_extraction(self):
        self.assertEqual(metrics.extract_str_answer('<think>reason</think><answer>CCO</answer>'), 'CCO')

    def test_prompt_compatibility(self):
        self.assertEqual(pipeline.mapping_list_qusetion_to_certain_form(['q'], None, False, False), ['q<think>'])
        self.assertEqual(pipeline.mapping_list_qusetion_to_certain_form(['q'], None, False, True), [{'prompt':'q<think>'}])

    def test_canonicalization(self):
        self.assertEqual(voting._canonicalize_smiles('OCC'), voting._canonicalize_smiles('CCO'))
        self.assertIsNone(voting._canonicalize_smiles('invalid!'))

    def test_task_scope(self):
        self.assertEqual(metrics.judge_task_from_answer_by_keyword(['[Reactant1]CCO']), 'retrosyn')
        self.assertEqual(metrics.judge_task_from_answer_by_keyword(['CCO']), 'nmr')

    def test_reactant_order(self):
        self.assertTrue(metrics.judge_single_retrosyn_answer("CCO.CN", "CN.OCC"))
        self.assertFalse(metrics.judge_single_retrosyn_answer("F_", "CCO"))

    def test_export_keeps_default_rope_theta_for_older_readers(self):
        import json
        import tempfile
        module = load_script("synreasoner_export_script", "train.py")
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"model_type": "qwen2", "rope_parameters": {"rope_type": "default", "rope_theta": 10000}}))
            tokenizer_config = Path(directory) / "tokenizer_config.json"
            tokenizer_config.write_text(json.dumps({"tokenizer_class": "TokenizersBackend", "eos_token": "EOS"}))
            self.assertIsNotNone(module.normalize_export_config(directory))
            tokenizer = json.loads(tokenizer_config.read_text())
            self.assertEqual(tokenizer["tokenizer_class"], "PreTrainedTokenizerFast")
            self.assertEqual(tokenizer["eos_token"], "EOS")
            exported = json.loads(config.read_text())
            self.assertEqual(exported["rope_theta"], 10000)
            self.assertIsNone(exported["rope_scaling"])
            self.assertEqual(exported["rope_parameters"]["rope_theta"], 10000)
            self.assertIsNone(module.normalize_export_config(directory))

    def test_training_import_has_no_side_effects(self):
        import importlib
        module = load_script("synreasoner_train_script", "train.py")
        self.assertTrue(callable(module.main))

if __name__ == '__main__':
    unittest.main()
