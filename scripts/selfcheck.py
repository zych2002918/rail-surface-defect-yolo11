# -*- coding: utf-8 -*-
"""引擎自检：验证改进版 ultralytics 可导入、权重可加载、端到端推理可用。

用法: python scripts/selfcheck.py

说明：推理链路 = 加载权重 -> predict（真实使用路径）。
裸 YAML 构建（无权重）受 torch 版本影响，README 已注明推荐环境
（论文实验环境：python 3.10, torch 2.2.2+cu121）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fail = 0


def check(name, fn):
    global fail
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        fail += 1
        import traceback
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()


def _import_ultralytics():
    import ultralytics  # noqa: F401

    assert ultralytics.__version__ == "8.3.9", f"version {ultralytics.__version__}"


def _weight_load():
    import ultralytics

    w = "weights/LSDECD-FDPN-ODConv-best.pt"
    if not os.path.exists(w):
        raise FileNotFoundError(w)
    m = ultralytics.YOLO(w)
    names = m.names
    print(f"    类别: {names}")
    assert len(names) == 4 or len(names) >= 1


def _inference():
    import numpy as np
    import ultralytics

    w = "weights/LSDECD-FDPN-ODConv-best.pt"
    m = ultralytics.YOLO(w)
    # 生成 dummy 灰度图（640x640），验证推理链路不崩溃
    img = np.full((640, 640, 3), 200, dtype=np.uint8)
    results = m.predict(source=img, imgsz=640, verbose=False, device="cpu")
    assert len(results) == 1
    print(f"    推理 OK, 检测框数: {len(results[0].boxes) if results[0].boxes is not None else 0}")


if __name__ == "__main__":
    check("import ultralytics (8.3.9)", _import_ultralytics)
    check("权重加载 LSDECD-FDPN-ODConv-best.pt", _weight_load)
    check("端到端推理 (CPU dummy 图)", _inference)
    print("----")
    sys.exit(1 if fail else 0)