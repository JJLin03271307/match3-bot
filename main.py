import yaml
from platform_adapters.windows import WinScreen, WinInput
from core.orchestrator import Orchestrator

if __name__ == "__main__":
    with open("games/sample_match3.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    screen = WinScreen(monitor=1)  # 游戏所在显示器
    inputc = WinInput(screen.bbox())  # 让输入映射知道该屏的物理像素
    orch = Orchestrator(screen, inputc, cfg)

    orch.run_one_level(level_id="L1")
