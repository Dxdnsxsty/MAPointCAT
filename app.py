# -*- coding: UTF-8 -*-
"""
@Project MAPointCAT
@File    app.py
@IDE     PyCharm
@Author  小帅天一(tianyi-yan@qq.com)
@Date    2026/5/8 09:50
"""
# -*- coding: utf-8 -*-
"""
MAPointCAT Streamlit Demo

功能：
1. 加载 ModelNet40 测试样本
2. 加载 PointCAT / MA / MA+CC 三类模型 checkpoint
3. 生成攻击样本
4. 展示同一个攻击点云下：
   - 原模型 PointCAT 分类失败
   - 现模型 MA+CC 分类成功
5. 支持自动搜索满足展示条件的样本
6. 支持普通单模型攻击展示
7. 支持批量鲁棒性评测
"""

import os
import sys
import random
import importlib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader

# ============================================================
# 0. 路径配置
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "model"))
sys.path.append(os.path.join(ROOT_DIR, "model", "classifier"))
sys.path.append(os.path.join(ROOT_DIR, "baselines"))

# ============================================================
# 1. 导入项目模块
# ============================================================

from data_utils.ModelNetDataLoader import ModelNetDataLoader

from baselines.attack.Noise.Jitter import JitterAttack
from baselines.attack.Drop.Drop import DropAttack
from baselines.attack.FGM.FGM import FGM, IFGM, MIFGM, PGD
from baselines.attack.CW.Perturb import CWPerturb
from baselines.attack.util.adv_utils import LogitsAdvLoss, CrossEntropyAdvLoss
from baselines.attack.util.clip_utils import ClipPointsL2
from baselines.attack.util.dist_utils import L2Dist

from baselines.defense.drop_points.SOR import SORDefense
from baselines.defense.drop_points.SRS import SRSDefense

def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

try:
    from baselines.defense.DUP_Net.DUP_Net import DUPNet

    HAS_DUPNET = True
except Exception:
    HAS_DUPNET = False

# ============================================================
# 2. 页面配置
# ============================================================

st.set_page_config(
    page_title="MAPointCAT 跨攻击泛化鲁棒性演示",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ MAPointCAT：跨攻击泛化鲁棒性的点云分类防御演示")

st.markdown(
    """
本系统用于展示 **3D 点云分类模型的跨攻击泛化鲁棒性增强效果**。

核心展示目标：

- 原模型：`PointCAT`
- 现模型：`MA+CC`
- 在同一个攻击点云下：
  - 原模型分类失败；
  - 现模型分类成功。
"""
)


# ============================================================
# 3. 通用工具函数
# ============================================================

def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu_id=0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(gpu_id)}")
    return torch.device("cpu")


def unwrap_logits(output):
    if isinstance(output, tuple):
        return output[1]
    return output


def strip_module_prefix(state_dict):
    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key

        new_state_dict[new_key] = value

    return new_state_dict


def read_modelnet40_class_names(data_path):
    file_path = os.path.join(data_path, "modelnet40_shape_names.txt")

    if not os.path.isfile(file_path):
        return [str(i) for i in range(40)]

    with open(file_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]

    return class_names


def format_label(label_id, class_names):
    label_id = int(label_id)

    if 0 <= label_id < len(class_names):
        return f"{label_id} / {class_names[label_id]}"

    return str(label_id)


def pc_to_numpy_bnc(points):
    """
    将点云转换为 [N, 3] numpy 格式。
    支持输入：
    - [N, 3]
    - [1, N, 3]
    - [1, 3, N]
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    arr = np.asarray(points)

    if arr.ndim == 3:
        arr = arr[0]

    if arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T

    return arr[:, :3]


def plot_point_cloud(points, title="Point Cloud", color=None):
    pts = pc_to_numpy_bnc(points)

    if color is None:
        color = pts[:, 2]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=color,
                    colorscale="Viridis",
                    opacity=0.9,
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
    )

    return fig


def get_confidence(logits, pred):
    prob = torch.softmax(logits, dim=-1)
    return float(prob[0, int(pred)].detach().cpu().item())


def compute_l2_distance(x_bnc, adv_bnc):
    """
    若 Drop 攻击导致点数不同，则不计算 L2。
    """
    if x_bnc.shape != adv_bnc.shape:
        return None

    diff = adv_bnc - x_bnc
    dist = torch.sqrt(torch.sum(diff ** 2, dim=[1, 2]))

    return dist.detach().cpu().numpy()


def is_demo_success(
        label,
        old_clean_pred,
        old_adv_pred,
        new_clean_pred,
        new_adv_pred,
        require_clean_correct=True,
):
    """
    展示成功条件：
    1. 原模型在攻击点云上分类失败；
    2. 现模型在同一个攻击点云上分类成功；
    3. 可选：两个模型在 clean 点云上都分类正确。
    """
    label = int(label)
    old_clean_pred = int(old_clean_pred)
    old_adv_pred = int(old_adv_pred)
    new_clean_pred = int(new_clean_pred)
    new_adv_pred = int(new_adv_pred)

    if require_clean_correct:
        if old_clean_pred != label:
            return False
        if new_clean_pred != label:
            return False

    old_failed = old_adv_pred != label
    new_success = new_adv_pred == label

    return old_failed and new_success


# ============================================================
# 4. 数据集与模型加载
# ============================================================

@st.cache_resource
def load_dataset_cached(data_path, input_point_nums, normal):
    dataset = ModelNetDataLoader(
        root=data_path,
        npoint=input_point_nums,
        split="test",
        normal_channel=normal,
    )

    return dataset


def build_pre_defense(pre_defense_name):
    if pre_defense_name == "None":
        return None

    if pre_defense_name == "SOR":
        return SORDefense(k=2, alpha=1.1)

    if pre_defense_name == "SRS":
        return SRSDefense(drop_num=500)

    if pre_defense_name == "DUP-Net":
        if not HAS_DUPNET:
            raise RuntimeError("当前环境无法导入 DUPNet，请检查 DUP_Net 依赖。")

        return DUPNet(
            sor_k=2,
            sor_alpha=1.1,
            npoint=1024,
            up_ratio=4,
        )

    raise NotImplementedError(pre_defense_name)


@st.cache_resource
def load_classifier_cached(
        checkpoint_path,
        defended_model,
        dataset_name,
        normal,
        pre_defense_name,
        gpu,
):
    device = get_device(gpu)

    if dataset_name == "ModelNet40":
        num_class = 40
    elif dataset_name == "ShapeNetPart":
        num_class = 16
    else:
        raise NotImplementedError(dataset_name)

    model_module = importlib.import_module(defended_model)

    use_pre_defense = pre_defense_name != "None"

    classifier = model_module.get_model(
        num_class,
        normal_channel=normal,
        use_pre_defense=use_pre_defense,
    )

    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = strip_module_prefix(state_dict)
    classifier.load_state_dict(state_dict, strict=False)

    classifier = classifier.to(device)
    classifier.eval()

    if use_pre_defense:
        pre_head = build_pre_defense(pre_defense_name)
        pre_head = pre_head.to(device)
        classifier.set_pre_head(pre_head)

    return classifier


# ============================================================
# 5. 攻击与预测函数
# ============================================================

def generate_target_labels(y, num_class=40):
    y_np = y.detach().cpu().numpy()
    targets = []

    for label in y_np:
        candidates = list(range(num_class))
        candidates.remove(int(label))
        targets.append(np.random.choice(candidates))

    return torch.tensor(targets, dtype=torch.long, device=y.device)


def build_adv_func(adv_func_name, mode, kappa):
    if adv_func_name == "logits":
        return LogitsAdvLoss(kappa=kappa, mode=mode)

    return CrossEntropyAdvLoss(mode=mode)


def attack_point_cloud(
        attack_name,
        source_model,
        target_model,
        x_bnc,
        y,
        num_class=40,
        targeted=False,
        adv_func_name="logits",
        kappa=0.0,
        delta=0.02,
        num_iter=10,
        mu=1.0,
        jitter_sigma=0.04,
        jitter_clip=0.16,
        drop_num=700,
        cw_lr=3e-3,
        cw_binary_step=3,
        cw_num_iter=30,
):
    """
    x_bnc: [B, N, 3]
    y: [B]
    """
    device = y.device

    source_model.eval()
    target_model.eval()

    batch_size, num_points, num_channels = x_bnc.shape

    if targeted:
        mode = "targeted"
        attack_target = generate_target_labels(y, num_class=num_class)
    else:
        mode = "untargeted"
        attack_target = y

    attack_info = {
        "targeted": targeted,
        "mode": mode,
        "attack_target": attack_target.detach().cpu().numpy().tolist(),
    }

    if attack_name == "None":
        return x_bnc.clone(), attack_info

    if attack_name == "Jitter":
        attacker = JitterAttack(
            target_model,
            sigma=jitter_sigma,
            clip=jitter_clip,
        )

        adv_np, acc_num = attacker(x_bnc.detach(), y)
        adv = torch.from_numpy(adv_np).float().to(device)

        attack_info["acc_num_after_attack"] = int(acc_num)

        return adv, attack_info

    if attack_name == "Drop":
        attacker = DropAttack(
            target_model,
            drop_num=drop_num,
        )

        adv_np, acc_num = attacker(x_bnc.detach(), y)
        adv = torch.from_numpy(adv_np).float().to(device)

        attack_info["acc_num_after_attack"] = int(acc_num)

        return adv, attack_info

    budget = delta * np.sqrt(num_points * num_channels)
    step_size = budget / float(max(num_iter, 1))

    adv_func = build_adv_func(
        adv_func_name=adv_func_name,
        mode=mode,
        kappa=kappa,
    )

    clip_func = ClipPointsL2(budget=budget)

    if attack_name == "FGM":
        attacker = FGM(
            source_model=source_model,
            target_model=target_model,
            adv_func=adv_func,
            budget=budget,
            dist_metric="l2",
        )

        adv_np, success_num = attacker.attack(
            x_bnc.detach(),
            attack_target,
            mode=mode,
        )

    elif attack_name == "IFGM":
        attacker = IFGM(
            source_model=source_model,
            target_model=target_model,
            adv_func=adv_func,
            clip_func=clip_func,
            budget=budget,
            step_size=step_size,
            num_iter=num_iter,
            dist_metric="l2",
        )

        adv_np, success_num = attacker.attack(
            x_bnc.detach(),
            attack_target,
            mode=mode,
        )

    elif attack_name == "MIFGM":
        attacker = MIFGM(
            source_model=source_model,
            target_model=target_model,
            adv_func=adv_func,
            clip_func=clip_func,
            budget=budget,
            step_size=step_size,
            num_iter=num_iter,
            mu=mu,
            dist_metric="l2",
        )

        adv_np, success_num = attacker.attack(
            x_bnc.detach(),
            attack_target,
            mode=mode,
        )

    elif attack_name == "PGD":
        attacker = PGD(
            source_model=source_model,
            target_model=target_model,
            adv_func=adv_func,
            clip_func=clip_func,
            budget=budget,
            step_size=step_size,
            num_iter=num_iter,
            dist_metric="l2",
        )

        adv_np, success_num = attacker.attack(
            x_bnc.detach(),
            attack_target,
            mode=mode,
        )

    elif attack_name == "C&W":
        dist_func = L2Dist()

        attacker = CWPerturb(
            source_model=source_model,
            target_model=target_model,
            adv_func=adv_func,
            dist_func=dist_func,
            attack_lr=cw_lr,
            init_weight=10.0,
            max_weight=80.0,
            binary_step=cw_binary_step,
            num_iter=cw_num_iter,
        )

        _, adv_np, success_num = attacker.attack(
            x_bnc.detach(),
            attack_target,
            mode=mode,
        )

    else:
        raise NotImplementedError(attack_name)

    adv = torch.from_numpy(adv_np).float().to(device)
    attack_info["success_num"] = int(success_num)

    return adv, attack_info


def predict(model, x_bnc):
    """
    x_bnc: [B, N, 3]
    model 输入: [B, 3, N]
    """
    x_bcn = x_bnc.transpose(1, 2).contiguous()

    with torch.no_grad():
        output = model(x_bcn)
        logits = unwrap_logits(output)
        pred = torch.argmax(logits, dim=-1)

    return pred, logits


# ============================================================
# 6. Sidebar 配置
# ============================================================

with st.sidebar:
    st.header("⚙️ 系统配置")

    data_path = st.text_input(
        "ModelNet40 数据路径",
        value="./data/modelnet40_normal_resampled/",
    )

    dataset_name = st.selectbox(
        "数据集",
        ["ModelNet40"],
        index=0,
    )

    defended_model = st.selectbox(
        "Backbone / Classifier",
        ["pointnet_cls", "pointnet2_cls_msg", "dgcnn", "curvenet"],
        index=0,
    )

    input_point_nums = st.number_input(
        "输入点数",
        min_value=128,
        max_value=4096,
        value=1024,
        step=128,
    )

    normal = st.checkbox(
        "使用 normal channel",
        value=False,
    )

    gpu = st.number_input(
        "GPU ID",
        min_value=0,
        max_value=8,
        value=0,
        step=1,
    )

    batch_size = st.number_input(
        "评测 batch size",
        min_value=1,
        max_value=128,
        value=16,
        step=1,
    )

    st.divider()
    st.header("🛡️ 模型 checkpoint")

    ckpt_pointcat = st.text_input(
        "PointCAT checkpoint",
        value="./log/pre_alter_pn/checkpoints/best-cls.pth",
    )

    ckpt_ma = st.text_input(
        "MA checkpoint",
        value="./log/exp_multi_no_cross/checkpoints/best-cls.pth",
    )

    ckpt_macc = st.text_input(
        "MA+CC checkpoint",
        value="./log/exp_multi_cross/checkpoints/best-cls.pth",
    )

    selected_defense_model_name = st.selectbox(
        "普通单模型展示使用的模型",
        ["PointCAT", "MA", "MA+CC"],
        index=2,
    )

    checkpoint_map = {
        "PointCAT": ckpt_pointcat,
        "MA": ckpt_ma,
        "MA+CC": ckpt_macc,
    }

    pre_defense_name = st.selectbox(
        "预处理防御",
        ["None", "SOR", "SRS", "DUP-Net"],
        index=0,
    )

    st.divider()
    st.header("⚔️ 攻击配置")

    attack_name = st.selectbox(
        "攻击方法",
        ["None", "Jitter", "Drop", "FGM", "IFGM", "MIFGM", "PGD", "C&W"],
        index=6,
    )

    # ========================================================
    # 默认攻击参数
    # 说明：
    # 即使某些参数当前攻击不用，也要给默认值。
    # 因为后面 attack_point_cloud(...) 调用时会统一传入这些变量。
    # ========================================================
    targeted = False
    adv_func_name = "logits"

    delta = 0.020
    num_iter = 10
    mu = 1.0

    jitter_sigma = 0.040
    jitter_clip = 0.160

    drop_num = 700

    cw_lr = 3e-3
    cw_binary_step = 3
    cw_num_iter = 30

    # ========================================================
    # 根据攻击方法动态显示对应参数
    # ========================================================

    if attack_name == "None":
        st.info("当前选择 None：不生成攻击样本，直接使用 clean 点云。")

    elif attack_name == "Jitter":
        st.markdown("#### Jitter 参数")

        jitter_sigma = st.slider(
            "Jitter sigma",
            min_value=0.001,
            max_value=0.100,
            value=0.040,
            step=0.001,
            format="%.3f",
            help="高斯噪声标准差，越大扰动越强。",
        )

        jitter_clip = st.slider(
            "Jitter clip",
            min_value=0.001,
            max_value=0.500,
            value=0.160,
            step=0.001,
            format="%.3f",
            help="噪声裁剪范围，限制单点最大扰动。",
        )

    elif attack_name == "Drop":
        st.markdown("#### Drop 参数")

        drop_num = st.slider(
            "Drop point number",
            min_value=1,
            max_value=1000,
            value=700,
            step=1,
            help="删除的点数量，越大攻击越强，但点云缺失越明显。",
        )

    elif attack_name == "FGM":
        st.markdown("#### FGM 参数")

        targeted = st.checkbox(
            "Targeted attack",
            value=False,
            help="是否使用目标攻击。未勾选时为 untargeted attack。",
        )

        adv_func_name = st.selectbox(
            "Adversarial loss",
            ["logits", "cross_entropy"],
            index=0,
            help="攻击损失函数。logits 通常更适合 margin-based 攻击。",
        )

        delta = st.slider(
            "扰动强度 delta",
            min_value=0.001,
            max_value=0.100,
            value=0.020,
            step=0.001,
            format="%.3f",
            help="FGM 的扰动预算系数，越大攻击越强。",
        )

    elif attack_name in ["IFGM", "PGD"]:
        st.markdown(f"#### {attack_name} 参数")

        targeted = st.checkbox(
            "Targeted attack",
            value=False,
            help="是否使用目标攻击。未勾选时为 untargeted attack。",
        )

        adv_func_name = st.selectbox(
            "Adversarial loss",
            ["logits", "cross_entropy"],
            index=0,
            help="攻击损失函数。logits 通常更适合 margin-based 攻击。",
        )

        delta = st.slider(
            "扰动强度 delta",
            min_value=0.001,
            max_value=0.100,
            value=0.020,
            step=0.001,
            format="%.3f",
            help="扰动预算系数，越大攻击越强。",
        )

        num_iter = st.slider(
            "迭代次数 num_iter",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="迭代攻击步数，越大攻击越强，但速度越慢。",
        )

    elif attack_name == "MIFGM":
        st.markdown("#### MIFGM 参数")

        targeted = st.checkbox(
            "Targeted attack",
            value=False,
            help="是否使用目标攻击。未勾选时为 untargeted attack。",
        )

        adv_func_name = st.selectbox(
            "Adversarial loss",
            ["logits", "cross_entropy"],
            index=0,
            help="攻击损失函数。logits 通常更适合 margin-based 攻击。",
        )

        delta = st.slider(
            "扰动强度 delta",
            min_value=0.001,
            max_value=0.100,
            value=0.020,
            step=0.001,
            format="%.3f",
            help="扰动预算系数，越大攻击越强。",
        )

        num_iter = st.slider(
            "迭代次数 num_iter",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="迭代攻击步数，越大攻击越强，但速度越慢。",
        )

        mu = st.slider(
            "MI-FGM momentum mu",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="动量系数，控制历史梯度的影响。",
        )

    elif attack_name == "C&W":
        st.markdown("#### C&W 参数")

        targeted = st.checkbox(
            "Targeted attack",
            value=False,
            help="是否使用目标攻击。未勾选时为 untargeted attack。",
        )

        adv_func_name = st.selectbox(
            "Adversarial loss",
            ["logits", "cross_entropy"],
            index=0,
            help="攻击损失函数。",
        )

        cw_lr = st.slider(
            "C&W learning rate",
            min_value=0.0001,
            max_value=0.0200,
            value=0.0030,
            step=0.0001,
            format="%.4f",
            help="C&W 优化学习率。",
        )

        cw_binary_step = st.slider(
            "C&W binary step",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="C&W 二分搜索步数。",
        )

        cw_num_iter = st.slider(
            "C&W num iter",
            min_value=5,
            max_value=200,
            value=30,
            step=5,
            help="C&W 每次优化的迭代次数。",
        )

    st.divider()
    st.header("🎬 论文展示配置")

    demo_old_model_name = st.selectbox(
        "原模型 / Baseline",
        ["PointCAT", "MA", "MA+CC"],
        index=0,
    )

    demo_new_model_name = st.selectbox(
        "现模型 / Proposed",
        ["PointCAT", "MA", "MA+CC"],
        index=2,
    )

    demo_max_search_samples = st.number_input(
        "自动寻找最大样本数",
        min_value=1,
        max_value=1000,
        value=200,
        step=10,
    )

    demo_require_clean_correct = st.checkbox(
        "要求两个模型在 clean 点云上都分类正确",
        value=True,
    )
    if st.button("重置自动搜索位置"):
        st.session_state.demo_search_start_idx = 0
        st.success("自动搜索位置已重置为 index = 0")

    if st.button("清除当前展示结果"):
        if "demo_result" in st.session_state:
            del st.session_state["demo_result"]
        st.success("已清除当前展示结果。")

# ============================================================
# 7. 加载数据集
# ============================================================

set_seed(2026)

device = get_device(gpu)

if not os.path.isdir(data_path):
    st.error(f"数据路径不存在：{data_path}")
    st.stop()

class_names = read_modelnet40_class_names(data_path)

try:
    dataset = load_dataset_cached(
        data_path=data_path,
        input_point_nums=input_point_nums,
        normal=normal,
    )
except Exception as e:
    st.error(f"加载数据集失败：{e}")
    st.stop()

st.success(f"已加载测试集，样本数：{len(dataset)}")

# ============================================================
# 8. 单样本选择
# ============================================================

st.header("🔍 单样本点云选择")

if "sample_idx_widget" not in st.session_state:
    st.session_state.sample_idx_widget = 0

# 新增：自动搜索起点，用于循环查找
# 作用：下一次点击“自动寻找成功展示样本”时，不再从 0 开始
if "demo_search_start_idx" not in st.session_state:
    st.session_state.demo_search_start_idx = 0

if "pending_sample_idx_widget" in st.session_state:
    st.session_state.sample_idx_widget = int(
        st.session_state["pending_sample_idx_widget"]
    )
    del st.session_state["pending_sample_idx_widget"]


def randomize_sample_idx():
    st.session_state.sample_idx_widget = random.randint(0, len(dataset) - 1)


sample_col_1, sample_col_2 = st.columns([1, 1])

with sample_col_1:
    st.number_input(
        "样本 index",
        min_value=0,
        max_value=len(dataset) - 1,
        step=1,
        key="sample_idx_widget",
    )

with sample_col_2:
    st.button(
        "随机选择样本",
        on_click=randomize_sample_idx,
    )

sample_idx = int(st.session_state.sample_idx_widget)

points_np, label_np = dataset[sample_idx]
label = int(label_np[0])

x_bnc = torch.from_numpy(points_np[:, :3]).float().unsqueeze(0).to(device)
y = torch.tensor([label], dtype=torch.long).to(device)

st.info(
    f"当前样本 index = {sample_idx}，"
    f"真实标签 = {format_label(label, class_names)}"
)

st.plotly_chart(
    plot_point_cloud(
        x_bnc,
        title="当前 Clean Point Cloud",
    ),
    width="stretch",
)

# ============================================================
# 9. 论文展示：原模型失败，现模型成功
# ============================================================

st.header("🎬 论文展示：原模型分类失败，现模型分类成功")

st.markdown(
    """
本模块使用 **原模型** 生成攻击点云，然后在同一个攻击点云上分别测试：

- 原模型；
- 现模型。

展示目标：

```text
原模型在攻击点云上分类失败
现模型在同一个攻击点云上分类成功
    """
)
demo_col_1, demo_col_2, demo_col_3 = st.columns(3)

with demo_col_1:
    run_demo_current = st.button(
        "使用当前样本生成对比展示",
        type="primary",
    )

with demo_col_2:
    run_demo_search = st.button(
        "自动寻找成功展示样本",
    )

with demo_col_3:
    demo_view_mode = st.radio(
        "展示视角",
        ["并排对比", "切换查看"],
        horizontal=True,
    )

st.caption(
    f"当前自动搜索起点 index = "
    f"{st.session_state.get('demo_search_start_idx', 0)}"
)

def run_pair_demo_for_index(sample_index):
    """
    对指定样本执行论文展示逻辑：
    1. 加载原模型和现模型；
    2. 使用原模型生成攻击点云；
    3. 在同一个攻击点云上分别测试原模型和现模型；
    4. 判断是否满足：原模型失败，现模型成功。
    """

    points_np_local, label_np_local = dataset[int(sample_index)]
    label_local = int(label_np_local[0])

    x_local = torch.from_numpy(points_np_local[:, :3]).float().unsqueeze(0).to(device)
    y_local = torch.tensor([label_local], dtype=torch.long).to(device)

    old_ckpt = checkpoint_map[demo_old_model_name]
    new_ckpt = checkpoint_map[demo_new_model_name]

    old_model = load_classifier_cached(
        checkpoint_path=old_ckpt,
        defended_model=defended_model,
        dataset_name=dataset_name,
        normal=normal,
        pre_defense_name=pre_defense_name,
        gpu=gpu,
    )

    new_model = load_classifier_cached(
        checkpoint_path=new_ckpt,
        defended_model=defended_model,
        dataset_name=dataset_name,
        normal=normal,
        pre_defense_name=pre_defense_name,
        gpu=gpu,
    )

    old_clean_pred, old_clean_logits = predict(old_model, x_local)
    new_clean_pred, new_clean_logits = predict(new_model, x_local)

    old_clean_pred_int = int(old_clean_pred.item())
    new_clean_pred_int = int(new_clean_pred.item())

    if attack_name == "None":
        raise RuntimeError("论文展示不能选择 None 攻击，请选择 PGD / IFGM / MIFGM 等。")

    # 核心逻辑：
    # 攻击样本只由原模型生成，然后用同一个攻击样本测试原模型和现模型。
    adv_local, attack_info = attack_point_cloud(
        attack_name=attack_name,
        source_model=old_model,
        target_model=old_model,
        x_bnc=x_local,
        y=y_local,
        num_class=40,
        targeted=targeted,
        adv_func_name=adv_func_name,
        kappa=0.0,
        delta=delta,
        num_iter=num_iter,
        mu=mu,
        jitter_sigma=jitter_sigma,
        jitter_clip=jitter_clip,
        drop_num=drop_num,
        cw_lr=cw_lr,
        cw_binary_step=cw_binary_step,
        cw_num_iter=cw_num_iter,
    )

    old_adv_pred, old_adv_logits = predict(old_model, adv_local)
    new_adv_pred, new_adv_logits = predict(new_model, adv_local)

    old_adv_pred_int = int(old_adv_pred.item())
    new_adv_pred_int = int(new_adv_pred.item())

    l2_dist = compute_l2_distance(x_local, adv_local)

    if l2_dist is None:
        l2_value = None
    else:
        l2_value = float(l2_dist[0])

    success = is_demo_success(
        label=label_local,
        old_clean_pred=old_clean_pred_int,
        old_adv_pred=old_adv_pred_int,
        new_clean_pred=new_clean_pred_int,
        new_adv_pred=new_adv_pred_int,
        require_clean_correct=demo_require_clean_correct,
    )

    return {
        "sample_idx": int(sample_index),
        "label": label_local,
        "x_bnc": x_local.detach().cpu(),
        "adv_bnc": adv_local.detach().cpu(),
        "attack_info": attack_info,
        "l2_dist": l2_value,
        "old_model_name": demo_old_model_name,
        "new_model_name": demo_new_model_name,
        "old_clean_pred": old_clean_pred_int,
        "old_adv_pred": old_adv_pred_int,
        "new_clean_pred": new_clean_pred_int,
        "new_adv_pred": new_adv_pred_int,
        "old_clean_conf": get_confidence(old_clean_logits, old_clean_pred_int),
        "old_adv_conf": get_confidence(old_adv_logits, old_adv_pred_int),
        "new_clean_conf": get_confidence(new_clean_logits, new_clean_pred_int),
        "new_adv_conf": get_confidence(new_adv_logits, new_adv_pred_int),
        "success": success,
        "attack_name": attack_name,
    }


def search_success_demo():
    """
    自动循环搜索满足论文展示条件的样本：
    原模型攻击后分类失败，现模型在同一攻击点云上分类成功。

    搜索逻辑：
    1. 第一次从 index = 0 开始；
    2. 如果本轮搜索到 index = 199，下次就从 index = 200 开始；
    3. 如果搜索到数据集末尾，会自动从 0 继续；
    4. 找到成功样本后，下次从成功样本的下一个 index 继续。
    """

    dataset_len = len(dataset)
    max_search = min(int(demo_max_search_samples), dataset_len)

    # 从 session_state 中读取本轮搜索起点
    start_idx = int(st.session_state.get("demo_search_start_idx", 0))

    # 防止数据集变化后 start_idx 越界
    if start_idx < 0 or start_idx >= dataset_len:
        start_idx = 0
        st.session_state.demo_search_start_idx = 0

    progress = st.progress(0)
    status = st.empty()

    for step in range(max_search):
        # 核心：循环搜索
        # 例如 start_idx=2400，dataset_len=2468
        # 则搜索顺序为 2400, 2401, ..., 2467, 0, 1, ...
        idx = (start_idx + step) % dataset_len

        status.text(
            f"正在循环搜索：{step + 1}/{max_search}，"
            f"当前 index = {idx}，本轮起点 = {start_idx}"
        )
        progress.progress((step + 1) / max_search)

        try:
            result = run_pair_demo_for_index(idx)
        except Exception as e:
            st.warning(f"样本 {idx} 运行失败，已跳过。原因：{e}")

            # 即使失败，也把下次搜索起点推进到下一个样本
            st.session_state.demo_search_start_idx = (idx + 1) % dataset_len
            continue

        # 每检查完一个样本，就更新下一次搜索起点
        st.session_state.demo_search_start_idx = (idx + 1) % dataset_len

        if result["success"]:
            status.success(
                f"找到成功展示样本：index = {idx}。"
                f"下次将从 index = {st.session_state.demo_search_start_idx} 继续搜索。"
            )
            return result

    status.error(
        f"本轮从 index = {start_idx} 开始，循环搜索了 {max_search} 个样本，"
        f"未找到满足条件的样本。下次将从 index = "
        f"{st.session_state.demo_search_start_idx} 继续搜索。"
    )

    return None


if run_demo_current:
    try:
        with st.spinner("正在生成当前样本对比展示..."):
            demo_result = run_pair_demo_for_index(sample_idx)

        st.session_state["demo_result"] = demo_result

    except Exception as e:
        st.error(f"生成对比展示失败：{e}")

if run_demo_search:
    try:
        with st.spinner("正在自动搜索成功展示样本..."):
            demo_result = search_success_demo()

        if demo_result is not None:
            st.session_state["demo_result"] = demo_result
            st.session_state["pending_sample_idx_widget"] = int(demo_result["sample_idx"])
            safe_rerun()

    except Exception as e:
        st.error(f"自动搜索失败：{e}")

if "demo_result" in st.session_state:
    demo = st.session_state["demo_result"]

    demo_label = int(demo["label"])
    old_model_name = demo["old_model_name"]
    new_model_name = demo["new_model_name"]

    st.subheader("展示结论")

    if demo["success"]:
        st.success(
            f"成功案例：原模型 {old_model_name} 分类失败，"
            f"现模型 {new_model_name} 分类成功。"
        )
    else:
        st.warning(
            "当前样本尚未满足“原模型失败，现模型成功”的条件，"
            "可以点击自动寻找。"
        )

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

    metric_col_1.metric(
        "样本 index",
        demo["sample_idx"],
    )

    metric_col_2.metric(
        "真实标签",
        format_label(demo_label, class_names),
    )

    metric_col_3.metric(
        "攻击方法",
        demo["attack_name"],
    )

    if demo["l2_dist"] is None:
        metric_col_4.metric("L2 Dist", "N/A")
    else:
        metric_col_4.metric("L2 Dist", f"{demo['l2_dist']:.4f}")

    result_df = pd.DataFrame(
        [
            {
                "模型": old_model_name,
                "Clean Pred": format_label(demo["old_clean_pred"], class_names),
                "Clean Conf": demo["old_clean_conf"],
                "Adv Pred": format_label(demo["old_adv_pred"], class_names),
                "Adv Conf": demo["old_adv_conf"],
                "Adv 是否正确": demo["old_adv_pred"] == demo_label,
            },
            {
                "模型": new_model_name,
                "Clean Pred": format_label(demo["new_clean_pred"], class_names),
                "Clean Conf": demo["new_clean_conf"],
                "Adv Pred": format_label(demo["new_adv_pred"], class_names),
                "Adv Conf": demo["new_adv_conf"],
                "Adv 是否正确": demo["new_adv_pred"] == demo_label,
            },
        ]
    )

    st.subheader("分类结果对比")

    st.dataframe(
        result_df.style.format(
            {
                "Clean Conf": "{:.4f}",
                "Adv Conf": "{:.4f}",
            }
        ),
        width="stretch",
    )

    if demo_view_mode == "并排对比":
        vis_col_1, vis_col_2, vis_col_3 = st.columns(3)

        with vis_col_1:
            st.plotly_chart(
                plot_point_cloud(
                    demo["x_bnc"],
                    title="Clean Point Cloud",
                ),
                width='stretch',
            )

        with vis_col_2:
            old_correct = demo["old_adv_pred"] == demo_label

            st.plotly_chart(
                plot_point_cloud(
                    demo["adv_bnc"],
                    title=f"原模型 {old_model_name}",
                ),
                width='stretch',
            )

            st.markdown(f"真实标签：**{format_label(demo_label, class_names)}**")
            st.markdown(
                f"预测标签：**{format_label(demo['old_adv_pred'], class_names)}**"
            )
            st.markdown(f"置信度：**{demo['old_adv_conf']:.4f}**")

            if old_correct:
                st.success("原模型分类正确")
            else:
                st.error("原模型分类失败")

        with vis_col_3:
            new_correct = demo["new_adv_pred"] == demo_label

            st.plotly_chart(
                plot_point_cloud(
                    demo["adv_bnc"],
                    title=f"现模型 {new_model_name}",
                ),
                width='stretch',
            )

            st.markdown(f"真实标签：**{format_label(demo_label, class_names)}**")
            st.markdown(
                f"预测标签：**{format_label(demo['new_adv_pred'], class_names)}**"
            )
            st.markdown(f"置信度：**{demo['new_adv_conf']:.4f}**")

            if new_correct:
                st.success("现模型分类成功")
            else:
                st.error("现模型分类失败")

    else:
        switch_model = st.radio(
            "切换模型查看同一个攻击点云的分类结果",
            [
                f"原模型 {old_model_name}",
                f"现模型 {new_model_name}",
            ],
            horizontal=True,
        )

        switch_col_1, switch_col_2 = st.columns([1, 2])

        with switch_col_1:
            st.markdown("### 当前视角")

            if switch_model.startswith("原模型"):
                current_model_name = old_model_name
                current_pred = demo["old_adv_pred"]
                current_conf = demo["old_adv_conf"]
                current_correct = current_pred == demo_label
            else:
                current_model_name = new_model_name
                current_pred = demo["new_adv_pred"]
                current_conf = demo["new_adv_conf"]
                current_correct = current_pred == demo_label

            st.markdown(f"**模型：** {current_model_name}")
            st.markdown(f"**真实标签：** {format_label(demo_label, class_names)}")
            st.markdown(f"**预测标签：** {format_label(current_pred, class_names)}")
            st.markdown(f"**置信度：** {current_conf:.4f}")

            if current_correct:
                st.success("分类成功")
            else:
                st.error("分类失败")

        with switch_col_2:
            st.plotly_chart(
                plot_point_cloud(
                    demo["adv_bnc"],
                    title=f"同一个攻击点云：{demo['attack_name']}",
                ),
                width='stretch',
            )

    with st.expander("展示样本详细信息"):
        st.json(
            {
                "sample_idx": int(demo["sample_idx"]),
                "label": int(demo["label"]),
                "label_name": class_names[int(demo["label"])]
                if int(demo["label"]) < len(class_names)
                else str(demo["label"]),
                "attack": demo["attack_name"],
                "old_model": old_model_name,
                "new_model": new_model_name,
                "old_clean_pred": int(demo["old_clean_pred"]),
                "old_adv_pred": int(demo["old_adv_pred"]),
                "new_clean_pred": int(demo["new_clean_pred"]),
                "new_adv_pred": int(demo["new_adv_pred"]),
                "success": bool(demo["success"]),
                "l2_dist": demo["l2_dist"],
            }
        )

# ============================================================
# 10. 普通单模型攻击展示
# ============================================================

st.header("🔬 普通单模型攻击与可视化")

run_single_attack = st.button(
    "生成单模型攻击样本并展示",
    type="secondary",
)

if run_single_attack:
    ckpt = checkpoint_map[selected_defense_model_name]

    try:
        model = load_classifier_cached(
            checkpoint_path=ckpt,
            defended_model=defended_model,
            dataset_name=dataset_name,
            normal=normal,
            pre_defense_name=pre_defense_name,
            gpu=gpu,
        )
    except Exception as e:
        st.error(f"加载模型失败：{e}")
        st.stop()

    pred_clean, logits_clean = predict(model, x_bnc)
    pred_clean_int = int(pred_clean.item())

    if attack_name == "None":
        adv_bnc = x_bnc.clone()
    else:
        with st.spinner(f"正在生成 {attack_name} 攻击样本..."):
            try:
                adv_bnc, _ = attack_point_cloud(
                    attack_name=attack_name,
                    source_model=model,
                    target_model=model,
                    x_bnc=x_bnc,
                    y=y,
                    num_class=40,
                    targeted=targeted,
                    adv_func_name=adv_func_name,
                    kappa=0.0,
                    delta=delta,
                    num_iter=num_iter,
                    mu=mu,
                    jitter_sigma=jitter_sigma,
                    jitter_clip=jitter_clip,
                    drop_num=drop_num,
                    cw_lr=cw_lr,
                    cw_binary_step=cw_binary_step,
                    cw_num_iter=cw_num_iter,
                )
            except Exception as e:
                st.error(f"生成攻击失败：{e}")
                st.stop()

    pred_adv, logits_adv = predict(model, adv_bnc)
    pred_adv_int = int(pred_adv.item())

    l2_dist = compute_l2_distance(x_bnc, adv_bnc)

    if l2_dist is None:
        l2_value = None
    else:
        l2_value = float(l2_dist[0])

    single_col_1, single_col_2, single_col_3, single_col_4 = st.columns(4)

    single_col_1.metric("真实标签", format_label(label, class_names))
    single_col_2.metric("Clean Pred", format_label(pred_clean_int, class_names))
    single_col_3.metric("Adv Pred", format_label(pred_adv_int, class_names))

    if l2_value is None:
        single_col_4.metric("L2 Dist", "N/A")
    else:
        single_col_4.metric("L2 Dist", f"{l2_value:.4f}")

    single_vis_col_1, single_vis_col_2 = st.columns(2)

    with single_vis_col_1:
        st.plotly_chart(
            plot_point_cloud(
                x_bnc,
                title="Original Point Cloud",
            ),
            width='stretch',
        )

    with single_vis_col_2:
        st.plotly_chart(
            plot_point_cloud(
                adv_bnc,
                title=f"Adversarial Point Cloud: {attack_name}",
            ),
            width='stretch',
        )

# ============================================================
# 11. 批量评测
# ============================================================

st.header("📊 不同防御方法在指定攻击下的性能评测")

eval_col_1, eval_col_2, eval_col_3 = st.columns(3)

with eval_col_1:
    eval_num_samples = st.number_input(
        "评测样本数",
        min_value=1,
        max_value=len(dataset),
        value=min(128, len(dataset)),
        step=1,
    )

with eval_col_2:
    eval_defenses = st.multiselect(
        "选择参与评测的模型",
        ["PointCAT", "MA", "MA+CC"],
        default=["PointCAT", "MA", "MA+CC"],
    )

with eval_col_3:
    run_eval = st.button(
        "开始批量评测",
        type="primary",
    )


def evaluate_model_under_attack(
        defense_name,
        checkpoint_path,
        pre_defense_name,
        dataset_obj,
        max_samples,
):
    model = load_classifier_cached(
        checkpoint_path=checkpoint_path,
        defended_model=defended_model,
        dataset_name=dataset_name,
        normal=normal,
        pre_defense_name=pre_defense_name,
        gpu=gpu,
    )

    loader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    total = 0
    clean_correct = 0
    adv_correct = 0
    attack_success = 0
    processed = 0

    progress = st.progress(0)
    status = st.empty()

    for batch in loader:
        points, target = batch

        if target.ndim > 1:
            target = target[:, 0]

        points = points[:, :, :3].float().to(device)
        target = target.long().to(device)

        remain = int(max_samples) - processed

        if remain <= 0:
            break

        if points.shape[0] > remain:
            points = points[:remain]
            target = target[:remain]

        current_batch_size = points.shape[0]

        pred_clean, _ = predict(model, points)
        clean_correct += (pred_clean == target).sum().item()

        if attack_name == "None":
            adv_points = points.clone()
            info = {}
        else:
            adv_points, info = attack_point_cloud(
                attack_name=attack_name,
                source_model=model,
                target_model=model,
                x_bnc=points,
                y=target,
                num_class=40,
                targeted=targeted,
                adv_func_name=adv_func_name,
                kappa=0.0,
                delta=delta,
                num_iter=num_iter,
                mu=mu,
                jitter_sigma=jitter_sigma,
                jitter_clip=jitter_clip,
                drop_num=drop_num,
                cw_lr=cw_lr,
                cw_binary_step=cw_binary_step,
                cw_num_iter=cw_num_iter,
            )

        pred_adv, _ = predict(model, adv_points)
        adv_correct += (pred_adv == target).sum().item()

        if targeted and attack_name not in ["None", "Jitter", "Drop"]:
            target_labels = torch.tensor(
                info["attack_target"],
                dtype=torch.long,
                device=device,
            )
            attack_success += (pred_adv == target_labels).sum().item()
        else:
            attack_success += (pred_adv != target).sum().item()

        total += current_batch_size
        processed += current_batch_size

        progress.progress(min(processed / int(max_samples), 1.0))
        status.text(f"{defense_name}: 已评测 {processed}/{int(max_samples)}")

        if processed >= int(max_samples):
            break

    clean_acc = clean_correct / max(total, 1)
    robust_acc = adv_correct / max(total, 1)
    success_rate = attack_success / max(total, 1)

    return {
        "Defense Model": defense_name,
        "Pre-defense": pre_defense_name,
        "Attack": attack_name,
        "Targeted": targeted,
        "Samples": total,
        "Clean Acc (%)": clean_acc * 100.0,
        "Robust Acc (%)": robust_acc * 100.0,
        "Attack Success / ASR (%)": success_rate * 100.0,
    }


if run_eval:
    if len(eval_defenses) == 0:
        st.warning("请至少选择一个模型。")
        st.stop()

    results = []

    for defense_name in eval_defenses:
        ckpt = checkpoint_map[defense_name]

        if not os.path.isfile(ckpt):
            st.error(f"{defense_name} checkpoint 不存在：{ckpt}")
            continue

        with st.spinner(f"正在评测 {defense_name} under {attack_name}..."):
            try:
                result = evaluate_model_under_attack(
                    defense_name=defense_name,
                    checkpoint_path=ckpt,
                    pre_defense_name=pre_defense_name,
                    dataset_obj=dataset,
                    max_samples=int(eval_num_samples),
                )

                results.append(result)

            except Exception as e:
                st.error(f"评测 {defense_name} 失败：{e}")

    if len(results) > 0:
        df = pd.DataFrame(results)

        st.subheader("评测结果")

        st.dataframe(
            df.style.format(
                {
                    "Clean Acc (%)": "{:.2f}",
                    "Robust Acc (%)": "{:.2f}",
                    "Attack Success / ASR (%)": "{:.2f}",
                }
            ),
            width='stretch',
        )

        csv = df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "下载 CSV 结果",
            data=csv,
            file_name=f"mapointcat_eval_{attack_name}.csv",
            mime="text/csv",
        )

        st.bar_chart(
            df.set_index("Defense Model")[["Clean Acc (%)", "Robust Acc (%)"]]
        )

# ============================================================
# 12. 方法说明
# ============================================================

with st.expander("📘 攻击方法说明"):
    st.markdown(
        """
| 攻击方法 | 说明 |
|---|---|
| None | 不攻击，测试 clean 点云 |
| Jitter | 对点云加入随机高斯扰动 |
| Drop | 删除部分点 |
| FGM | 单步梯度攻击 |
| IFGM | 迭代梯度攻击 |
| MIFGM | 动量迭代梯度攻击 |
| PGD | 投影梯度攻击 |
| C&W | 基于优化的 C&W 点云扰动攻击 |
"""
    )

with st.expander("📘 防御模型说明"):
    st.markdown(
        """
| 模型 | 说明 |
|---|---|
| PointCAT | 原 baseline，Contrastive Adversarial Training |
| MA | Multi-Attack Training |
| MA+CC | Multi-Attack Training + Cross-Attack Consistency Loss |
| SOR | Statistical Outlier Removal 预处理防御 |
| SRS | Simple Random Sampling 预处理防御 |
| DUP-Net | SOR + PU-Net 上采样防御 |
"""
    )

with st.expander("📌 推荐展示设置"):
    st.markdown(
        """
建议论文或答辩展示时使用：

```text
原模型 / Baseline: PointCAT
现模型 / Proposed: MA+CC
攻击方法: PGD / IFGM / MIFGM
delta: 0.020 ~ 0.040
num_iter: 10 ~ 20
自动寻找最大样本数: 200 ~ 500
如果自动搜索找不到成功样本，可以尝试：
    1. 增大 delta；
    2. 增大 num_iter；
    3. 切换 PGD / IFGM / MIFGM；
    4. 临时取消“要求两个模型在 clean 点云上都分类正确”。 
    """
    )
