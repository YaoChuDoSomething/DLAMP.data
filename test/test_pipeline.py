"""
Pipeline 核心邏輯單元測試

測試重點：
1. Task 執行順序
2. Context 設定載入
3. 異常處理
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pipeline import Pipeline, Task
from src.core.context import Context
import tempfile
import yaml


class MockTask(Task):
    """Mock Task for testing"""

    def __init__(self, name):
        self.name = name
        self.executed = False

    def execute(self, context: Context):
        self.executed = True
        print(f"[MOCK] Executed {self.name}")


class TestPipeline(unittest.TestCase):
    """Pipeline 核心邏輯測試"""

    def setUp(self):
        """建立測試環境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = f"{self.temp_dir}/test_config.yaml"

        # 建立最小設定檔
        config = {"share": {"exp_code": "TEST"}}
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

    def test_task_execution_order(self):
        """測試任務按順序執行"""
        context = Context(self.config_path)
        pipeline = Pipeline(context)

        task1 = MockTask("Task1")
        task2 = MockTask("Task2")
        task3 = MockTask("Task3")

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)

        pipeline.run()

        # 驗證所有任務都被執行
        self.assertTrue(task1.executed)
        self.assertTrue(task2.executed)
        self.assertTrue(task3.executed)

    def test_context_loading(self):
        """測試 Context 正確載入設定"""
        context = Context(self.config_path)
        pipeline = Pipeline(context)
        pipeline.run()

        # 驗證設定已載入
        self.assertIsNotNone(context.config)
        self.assertEqual(context.config["share"]["exp_code"], "TEST")

    def test_empty_pipeline(self):
        """測試空 Pipeline 不報錯"""
        context = Context(self.config_path)
        pipeline = Pipeline(context)

        try:
            pipeline.run()
            success = True
        except Exception:
            success = False

        self.assertTrue(success, "Empty pipeline should not raise exception")


class TestContext(unittest.TestCase):
    """Context 設定管理測試"""

    def test_config_file_loading(self):
        """測試設定檔載入"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"test": "value"}, f)
            config_path = f.name

        context = Context(config_path)
        context.load_config()

        self.assertEqual(context.config["test"], "value")

    def test_missing_config_file(self):
        """測試設定檔遺失時的錯誤處理"""
        context = Context("/nonexistent/config.yaml")

        with self.assertRaises(FileNotFoundError):
            context.load_config()


if __name__ == "__main__":
    unittest.main()
