# export_torchscript.py

import os
import torch
from models.common import DetectMultiBackend

def get_latest_best_pt(base_dir='runs/train'):
    """
    自动查找 runs/train 文件夹下编号最大的 expX/weights/best.pt 路径。
    """
    exps = [d for d in os.listdir(base_dir) if d.startswith('exp') and os.path.isdir(os.path.join(base_dir, d))]
    if not exps:
        raise FileNotFoundError(f"No exp folders found in {base_dir}")

    # 提取exp编号，按数字排序
    exps_sorted = sorted(
        exps,
        key=lambda x: int(x.replace('exp', '')) if x.replace('exp', '').isdigit() else -1
    )
    latest_exp = exps_sorted[-1]
    best_pt_path = os.path.join(base_dir, latest_exp, 'weights', 'best.pt')

    if not os.path.exists(best_pt_path):
        raise FileNotFoundError(f"best.pt not found at {best_pt_path}")

    print(f"[INFO] Using weights from: {best_pt_path}")
    return best_pt_path

def export_torchscript(weights_path, export_path='best_scripted.pth'):
    """
    将 YOLOv5 权重导出为 TorchScript 格式。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 YOLOv5 模型
    model_wrapper = DetectMultiBackend(weights_path, device=device)
    model = model_wrapper.model
    model.eval()

    # 用 dummy 输入进行 trace
    dummy_input = torch.randn(1, 3, 640, 640, device=device)
    scripted_model = torch.jit.trace(model, dummy_input)

    # 保存 TorchScript 文件
    scripted_model.save(export_path)
    print(f"[INFO] TorchScript saved at: {export_path}")

if __name__ == "__main__":
    weights = get_latest_best_pt(base_dir='runs/train')
    export_torchscript(weights_path=weights, export_path='best_scripted.pth')
