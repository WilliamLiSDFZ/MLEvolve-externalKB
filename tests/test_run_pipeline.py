"""Check the CLI's pipeline and signal wiring without importing the ML stack."""

import ast
from pathlib import Path
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock


class RunPipelineTests(unittest.TestCase):
    def test_pipeline_and_sigterm_restore_handler_and_clean_up(self):
        source = Path(__file__).resolve().parents[1] / "run.py"
        run_function = next(node for node in ast.parse(source.read_text()).body
                            if isinstance(node, ast.FunctionDef) and node.name == "run")
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                cfg = SimpleNamespace(torch_hub_dir=None, exp_name="test", workspace_dir="workspace",
                                      coldstart=SimpleNamespace(use_coldstart=False), exec={},
                                      agent=SimpleNamespace(steps=3))
                journal = MagicMock()
                journal.__len__.return_value = 1
                interpreter = Mock()
                agent = Mock()
                old_handler = object()
                signals = SimpleNamespace(SIGTERM=signal.SIGTERM,
                                          signal=Mock(return_value=old_handler))
                save = Mock()

                def pipeline(actual_agent, actual_interpreter, actual_cfg, callback, save_callback):
                    self.assertIs(actual_agent, agent)
                    self.assertIs(actual_interpreter, interpreter)
                    self.assertIs(actual_cfg, cfg)
                    self.assertIs(callback("code", "node", True), interpreter.run.return_value)
                    interpreter.run.assert_called_once_with("code", "node", True)
                    save_callback()
                    if interrupted:
                        handler = signals.signal.call_args.args[1]
                        handler(signal.SIGTERM, None)

                namespace = {
                    "load_cfg": lambda: cfg, "set_global_seed": Mock(), "setup_logging": Mock(),
                    "load_task_desc": Mock(), "Status": MagicMock(), "prep_agent_workspace": Mock(),
                    "atexit": SimpleNamespace(register=Mock()), "shutil": Mock(),
                    "Journal": lambda: journal, "Agent": Mock(return_value=agent),
                    "Interpreter": Mock(return_value=interpreter),
                    "OmegaConf": SimpleNamespace(to_container=lambda value: value),
                    "signal": signals, "run_search_pipeline": pipeline, "save_run": save,
                }
                cfg.agent.seed = 42
                exec(compile(ast.Module(body=[run_function], type_ignores=[]), str(source), "exec"), namespace)
                if interrupted:
                    with self.assertRaises(SystemExit) as raised:
                        namespace["run"]()
                    self.assertEqual(raised.exception.code, 128 + signal.SIGTERM)
                else:
                    namespace["run"]()
                save.assert_called_once_with(cfg, journal)
                interpreter.cleanup_session.assert_called_once_with(-1)
                self.assertEqual(signals.signal.call_count, 2)
                signals.signal.assert_called_with(signal.SIGTERM, old_handler)


if __name__ == "__main__":
    unittest.main()
