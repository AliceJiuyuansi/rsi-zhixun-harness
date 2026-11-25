"""SDK receipt handling and replay parsing. No real host or model calls."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "harness")]
from contest_submission import _model_prompt
from src.harness.model_receipt import decide_with_receipt
from src.harness.round_controller import parse_verdict


PARTIAL = '... {"decision":"train","epochs":2,"lr":0.001,"reason":"truncated'
RULE = {"action": "finetune", "epochs": 3, "lr": .0015, "rule": "new_stage", "signals": {}}


class ReceiptTests(unittest.TestCase):
    def decide(self, text, finish=None, channel="content", **quota):
        response = {"choices": [{"message": {channel: text}, "finish_reason": finish}]}
        model = Mock(return_value=SimpleNamespace(response=response))
        decision, summary, diagnostics = decide_with_receipt(SimpleNamespace(model=model, **quota),
                                                             "prompt", RULE, {}, 4)
        model.assert_called_once_with([{"role": "user", "content": "prompt"}],
                                      max_tokens=diagnostics["max_tokens"], temperature=0)
        return decision, summary, diagnostics

    def test_offline_parser_recovers_only_explicit_json(self):
        self.assertEqual(parse_verdict(PARTIAL), {"action": "train", "epochs": 2.0, "lr": .001,
                                                 "reason": "truncated_json_recovered"})
        for text in ('continue training please', '"decision":"hold"', '{"reason":"train',
                     '{"decision":"tra', '{"action":"do something"', '{"decision":"train" garbage',
                     '{"decision":"train","epochs":1e', '{"decision":"train","lr":0.00',
                     '{"action":"hold","decision":"train"', '{"action":"hold","action":"train"'):
            with self.subTest(text=text):
                self.assertIsNone(parse_verdict(text))

    def test_live_partial_requires_length(self):
        for finish in (None, 'stop', 'content_filter', 'length'):
            with self.subTest(finish=finish):
                decision, summary, diag = self.decide(PARTIAL, finish, 'reasoning_content')
                self.assertEqual(diag['text_chars'], len(PARTIAL))
                self.assertEqual(diag['text_source'], 'reasoning_content')
                self.assertNotIn(PARTIAL, summary)
                self.assertNotIn('reasoning_content', summary)
                if finish == 'length':
                    self.assertEqual((decision['source'], decision['epochs'], decision['lr']), ('llm', 2, .001))
                    self.assertEqual(diag['status'], 'truncated_json_recovered')
                    self.assertTrue(diag['changed_rule'])
                else:
                    self.assertEqual((decision['model_action'], decision['source']), ('unavailable', 'rule'))
                    self.assertEqual((summary, diag['status'], diag['changed_rule']), ('', 'invalid_json', False))

    def test_recovered_agreement_does_not_claim_change(self):
        decision, _, diag = self.decide('{"action":"train",', 'length')
        self.assertEqual((decision['epochs'], decision['lr'], decision['source']), (3, .0015, 'rule'))
        self.assertFalse(diag['changed_rule'])

    def test_no_prose_from_either_channel_reaches_state(self):
        for channel in ('reasoning_content', 'content'):
            for text in ('private prose only', 'private prose: {"action":"hold","reason":"private reason"}'):
                with self.subTest(channel=channel, valid='{' in text):
                    result = self.decide(text, 'length', channel)
                    self.assertNotIn('private', json.dumps(result))
                    self.assertEqual(result[2]['status'], 'valid_json' if '{' in text else 'invalid_json')

    def test_limits_and_prompt(self):
        prompt = _model_prompt({'round': 4})
        self.assertTrue(prompt.startswith('/no_think\n'))
        self.assertTrue(prompt.endswith('{"round":4}'))
        for quota, expected in (({}, 768), ({'max_output_tokens': 1024}, 768),
                                ({'max_output_tokens': 96}, 96), ({'max_output_tokens': 0}, 768)):
            with self.subTest(quota=quota):
                _, _, diag = self.decide('', 'length', **quota)
                self.assertEqual((diag['max_tokens'], diag['status']), (expected, 'empty'))

    def test_compact_prompt_retains_decision_inputs(self):
        payload = {'round': 4, 'history': [{'round': 3, 'gain': 0.01}],
                   'history_evidence': 'duplicate history', 'rule_budget': RULE}
        prompt = _model_prompt(payload, compact=True)
        self.assertTrue(prompt.startswith('/no_think\n'))
        self.assertLess(len(prompt), len(_model_prompt(payload)))
        state = json.loads(prompt.splitlines()[-1])
        self.assertNotIn('history_evidence', state)
        self.assertEqual(state['history'], payload['history'])
        self.assertEqual(state['rule_budget'], {k: RULE[k] for k in ('action', 'epochs', 'lr', 'rule')})

    def test_non_thinking_call_respects_host_quota_then_uses_rule_on_empty(self):
        response = {'choices': [{'message': {'content': ''}, 'finish_reason': 'length'}]}
        for quota, expected in ((1024, 256), (96, 96)):
            with self.subTest(quota=quota):
                model = Mock(return_value=SimpleNamespace(response=response))
                rnd = SimpleNamespace(model=model, max_output_tokens=quota)
                decision, _, diag = decide_with_receipt(
                    rnd, 'prompt', RULE, {}, 4, max_tokens=256, enable_thinking=False)
                model.assert_called_once_with([{'role': 'user', 'content': 'prompt'}],
                                              max_tokens=expected, temperature=0, enable_thinking=False)
                self.assertEqual((diag['status'], diag['max_tokens']), ('empty', expected))
                self.assertEqual((decision['source'], decision['epochs'], decision['lr']), ('rule', 3, .0015))


if __name__ == '__main__':
    unittest.main()
