# -*- coding: utf-8 -*-
"""
MAPointCAT Streamlit Demo System

功能：
1. 加载 ModelNet40 点云样本
2. 加载不同防御模型 checkpoint：PointCAT / MA / MA+CC
3. 生成攻击样本：Jitter / Drop / FGM / IFGM / MI-FGM / PGD / C&W
4. 展示原始点云和攻击后点云
5. 评测不同防御方法在指定攻击下的性能

运行：
streamlit run streamlit_app.py
"""

import os
import sys
import copy
import random
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================
# 0. 路径配置
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "model"))
sys.path.append(os.path.join(ROOT_DIR, "model/classifier"))
sys.path.append(os.path.join(ROOT_DIR, "baselines"))

# ============================================================
# 1. 导入你仓库中的模块
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

try:
    from baselines.defense.DUP_Net.DUP_Net import DUPNet

    HAS_DUPNET = True
except Exception:
    HAS_DUPNET = False

# ============================================================
# 2. Streamlit 页面配置
# ============================================================

st.set_page_config(
    page_title="MAPointCAT 点云鲁棒性演示系统",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ MAPointCAT：面向 3D 点云分类模型的攻击与防御演示系统")

st.markdown(
    """
本系统用于展示和评测 **3D 点云分类模型的攻击与防御方法**。

支持功能：

- 加载 ModelNet40 点云样本；
- 展示原始点云和攻击后的点云；
- 生成不同攻击样本；
- 加载不同防御模型：`PointCAT / MA / MA+CC`；
- 可选预处理防御：`SOR / SRS / DUP-Net`；
- 测评不同防御方法在指定攻击下的分类准确率和攻击成功率。
"""
)


# ============================================================
# 3. 工具函数
# ============================================================

def set_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu_id=0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def unwrap_logits(output):
    """
    适配 PointNet 返回格式：
    pointnet_cls.forward 返回 (h, logits)
    有些模型可能直接返回 logits。
    """
    if isinstance(output, tuple):
        return output[1]
    return output


def unwrap_feat_logits(output):
    """
    返回 feature 和 logits。
    如果模型只返回 logits，则 feature 为 None。
    """
    if isinstance(output, tuple):
        return output[0], output[1]
    return None, output


def strip_module_prefix(state_dict):
    """
    兼容 DataParallel 保存的 module.xxx 权重。
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def read_modelnet40_class_names(data_path):
    file_path = os.path.join(data_path, "modelnet40_shape_names.txt")
    if not os.path.isfile(file_path):
        return [str(i) for i in range(40)]
    with open(file_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines()]
    return names


def pc_to_numpy_bnc(points):
    """
    输入可能是：
    [N, 3]
    [1, N, 3]
    [1, 3, N]
    输出 [N, 3]
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
    """
    使用 Plotly 展示 3D 点云。
    points: [N, 3]
    """
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


def make_default_args(
        dataset,
        data_path,
        defended_model,
        input_point_nums,
        normal,
        batch_size,
        gpu,
):
    """
    构造和你原始 tester / solver 类似的 args。
    """
    device = get_device(gpu)

    args = SimpleNamespace()
    args.dataset = dataset
    args.data_path = data_path
    args.defended_model = defended_model
    args.input_point_nums = input_point_nums
    args.normal = normal
    args.batch_size = batch_size
    args.gpu = gpu
    args.device = device
    args.use_multi_gpu = False

    # 攻击相关默认参数
    args.adv_func = "logits"
    args.kappa = 0.0
    args.budget = 0.08
    args.num_iter = 50
    args.mu = 1.0
    args.attack_lr = 1e-2
    args.binary_step = 5
    args.num_iter_cw = 50

    # defense / generator 相关参数
    args.use_pre_defense = False
    args.pre_defense = None

    # AutoEncoder / PointCAT generator 可能需要的字段，保留
    args.decoder_type = "normal_conv"
    args.DEGREE = [1, 2, 2, 2, 2, 2, 64]
    args.D_FEAT = [3, 64, 128, 256, 256, 512]
    args.G_FEAT = [96, 256, 256, 256, 128, 128, 128, 3]
    args.loop_non_linear = False
    args.support = 10

    return args


# ============================================================
# 4. 模型与数据加载
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
    """
    构造预处理防御。
    注意：DUP-Net 依赖 PU-Net 权重路径，原仓库中有硬编码路径，建议按本文后面的说明修改。
    """
    if pre_defense_name == "None":
        return None

    if pre_defense_name == "SOR":
        return SORDefense(k=2, alpha=1.1)

    if pre_defense_name == "SRS":
        return SRSDefense(drop_num=500)

    if pre_defense_name == "DUP-Net":
        if not HAS_DUPNET:
            raise RuntimeError("当前环境无法导入 DUPNet，请检查 DUP_Net 相关依赖或权重路径。")
        return DUPNet(sor_k=2, sor_alpha=1.1, npoint=1024, up_ratio=4)

    raise NotImplementedError(pre_defense_name)


@st.cache_resource
def load_classifier_cached(
        checkpoint_path,
        defended_model,
        dataset,
        normal,
        pre_defense_name,
        gpu,
):
    """
    加载分类模型。
    checkpoint_path 可以是 PointCAT / MA / MA+CC 的 checkpoint。
    """
    device = get_device(gpu)

    if dataset == "ModelNet40":
        num_class = 40
    elif dataset == "ShapeNetPart":
        num_class = 16
    else:
        raise NotImplementedError(dataset)

    MODEL = importlib.import_module(defended_model)

    use_pre_defense = pre_defense_name != "None"

    classifier = MODEL.get_model(
        num_class,
        normal_channel=normal,
        use_pre_defense=use_pre_defense,
    )

    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        state_dict = strip_module_prefix(state_dict)
        classifier.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")

    classifier = classifier.to(device)
    classifier.eval()

    if use_pre_defense:
        pre_head = build_pre_defense(pre_defense_name)
        pre_head = pre_head.to(device)
        classifier.set_pre_head(pre_head)

    return classifier


# ============================================================
# 5. 攻击样本生成
# ============================================================

def generate_target_labels(y, num_class=40):
    """
    为 targeted attack 随机生成不同于真实标签的目标标签。
    """
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
    else:
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
    生成攻击样本。

    参数：
    x_bnc: torch.Tensor [B, N, 3]
    y: torch.Tensor [B]
    返回：
    adv_bnc: torch.Tensor [B, N, 3] 或 Drop 后 [B, N-drop, 3]
    attack_info: dict
    """
    device = y.device

    source_model.eval()
    target_model.eval()

    B, N, C = x_bnc.shape

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

    # --------------------------------------------------------
    # Random jitter
    # --------------------------------------------------------
    if attack_name == "Jitter":
        attacker = JitterAttack(target_model, sigma=jitter_sigma, clip=jitter_clip)
        adv_np, acc_num = attacker(x_bnc.detach(), y)
        adv = torch.from_numpy(adv_np).float().to(device)
        attack_info["acc_num_after_attack"] = int(acc_num)
        return adv, attack_info

    # --------------------------------------------------------
    # Random drop
    # --------------------------------------------------------
    if attack_name == "Drop":
        attacker = DropAttack(target_model, drop_num=drop_num)
        adv_np, acc_num = attacker(x_bnc.detach(), y)
        adv = torch.from_numpy(adv_np).float().to(device)
        attack_info["acc_num_after_attack"] = int(acc_num)
        return adv, attack_info

    # --------------------------------------------------------
    # Gradient-based attacks
    # --------------------------------------------------------
    budget = delta * np.sqrt(N * C)
    step_size = budget / float(max(num_iter, 1))

    adv_func = build_adv_func(adv_func_name, mode=mode, kappa=kappa)
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
    model 输入为 [B, 3, N]
    """
    x_bcn = x_bnc.transpose(1, 2).contiguous()
    with torch.no_grad():
        output = model(x_bcn)
        logits = unwrap_logits(output)
        pred = torch.argmax(logits, dim=-1)
    return pred, logits


def compute_l2_distance(x_bnc, adv_bnc):
    """
    对 Drop 攻击，点数不同，不计算严格 L2。
    """
    if x_bnc.shape != adv_bnc.shape:
        return None

    diff = adv_bnc - x_bnc
    dist = torch.sqrt(torch.sum(diff ** 2, dim=[1, 2]))
    return dist.detach().cpu().numpy()


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

    normal = st.checkbox("使用 normal channel", value=False)

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

    st.header("🛡️ 防御模型 checkpoint")

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
        "单样本展示使用的训练防御模型",
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

    targeted = st.checkbox("Targeted attack", value=False)

    adv_func_name = st.selectbox(
        "Adversarial loss",
        ["logits", "cross_entropy"],
        index=0,
    )

    delta = st.slider(
        "扰动强度 delta",
        min_value=0.001,
        max_value=0.100,
        value=0.020,
        step=0.001,
        format="%.3f",
    )

    num_iter = st.slider(
        "迭代次数 num_iter",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )

    mu = st.slider(
        "MI-FGM momentum mu",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
    )

    jitter_sigma = st.slider(
        "Jitter sigma",
        min_value=0.001,
        max_value=0.100,
        value=0.040,
        step=0.001,
        format="%.3f",
    )

    jitter_clip = st.slider(
        "Jitter clip",
        min_value=0.001,
        max_value=0.500,
        value=0.160,
        step=0.001,
        format="%.3f",
    )

    drop_num = st.slider(
        "Drop point number",
        min_value=1,
        max_value=1000,
        value=700,
        step=1,
    )

    st.caption("C&W 较慢，建议展示时降低 binary step 和 iter。")

    cw_binary_step = st.slider(
        "C&W binary step",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )

    cw_num_iter = st.slider(
        "C&W num iter",
        min_value=5,
        max_value=200,
        value=30,
        step=5,
    )

# ============================================================
# 7. 加载数据
# ============================================================

set_seed(2022)
device = get_device(gpu)

if not os.path.isdir(data_path):
    st.error(f"数据路径不存在：{data_path}")
    st.stop()

class_names = read_modelnet40_class_names(data_path)

try:
    dataset = load_dataset_cached(data_path, input_point_nums, normal)
except Exception as e:
    st.error(f"加载数据集失败：{e}")
    st.stop()

st.success(f"已加载测试集，样本数：{len(dataset)}")

# ============================================================
# 8. 单样本展示
# ============================================================

st.header("🔍 单样本攻击与可视化")

col_sample_1, col_sample_2, col_sample_3 = st.columns([1, 1, 1])

with col_sample_1:
    sample_idx = st.number_input(
        "样本 index",
        min_value=0,
        max_value=len(dataset) - 1,
        value=0,
        step=1,
    )

with col_sample_2:
    random_sample = st.button("随机选择样本")

with col_sample_3:
    run_single_attack = st.button("生成攻击样本并展示", type="primary")

if random_sample:
    sample_idx = random.randint(0, len(dataset) - 1)

points_np, label_np = dataset[int(sample_idx)]
label = int(label_np[0])

st.info(f"当前样本 index = {sample_idx}, label = {label} / {class_names[label] if label < len(class_names) else label}")

x_bnc = torch.from_numpy(points_np[:, :3]).float().unsqueeze(0).to(device)
y = torch.tensor([label], dtype=torch.long).to(device)

if run_single_attack:
    ckpt = checkpoint_map[selected_defense_model_name]

    try:
        model = load_classifier_cached(
            checkpoint_path=ckpt,
            defended_model=defended_model,
            dataset=dataset_name,
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
        attack_info = {}
    else:
        with st.spinner(f"正在生成 {attack_name} 攻击样本..."):
            try:
                adv_bnc, attack_info = attack_point_cloud(
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
                    cw_lr=3e-3,
                    cw_binary_step=cw_binary_step,
                    cw_num_iter=cw_num_iter,
                )
            except Exception as e:
                st.error(f"生成攻击失败：{e}")
                st.stop()

    pred_adv, logits_adv = predict(model, adv_bnc)
    pred_adv_int = int(pred_adv.item())

    l2_dist = compute_l2_distance(x_bnc, adv_bnc)

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

    metric_col_1.metric(
        "真实标签",
        f"{label} / {class_names[label] if label < len(class_names) else label}",
    )
    metric_col_2.metric(
        "Clean Pred",
        f"{pred_clean_int} / {class_names[pred_clean_int] if pred_clean_int < len(class_names) else pred_clean_int}",
    )
    metric_col_3.metric(
        "Adv Pred",
        f"{pred_adv_int} / {class_names[pred_adv_int] if pred_adv_int < len(class_names) else pred_adv_int}",
    )

    if l2_dist is None:
        metric_col_4.metric("L2 Dist", "N/A")
    else:
        metric_col_4.metric("L2 Dist", f"{float(l2_dist[0]):.4f}")

    if targeted and attack_name not in ["None", "Jitter", "Drop"]:
        attack_target = attack_info.get("attack_target", [None])[0]
        st.warning(f"Targeted attack target label = {attack_target}")

    col_vis_1, col_vis_2 = st.columns(2)

    with col_vis_1:
        st.plotly_chart(
            plot_point_cloud(x_bnc, title="Original Point Cloud"),
            use_container_width=True,
        )

    with col_vis_2:
        st.plotly_chart(
            plot_point_cloud(adv_bnc, title=f"Adversarial Point Cloud: {attack_name}"),
            use_container_width=True,
        )

# ============================================================
# 9. 批量评测
# ============================================================

st.header("📊 不同防御方法在指定攻击下的性能评测")

st.markdown(
    """
这里会对选择的攻击方法进行批量评测，并比较不同训练防御模型：

- PointCAT
- MA
- MA+CC

指标：

- `Clean Acc`：无攻击准确率；
- `Robust Acc`：攻击后准确率；
- `Untargeted ASR`：非目标攻击成功率，约等于 `1 - Robust Acc`；
- `Targeted Success`：目标攻击成功率。
"""
)

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
        "选择参与评测的训练防御模型",
        ["PointCAT", "MA", "MA+CC"],
        default=["PointCAT", "MA", "MA+CC"],
    )

with eval_col_3:
    run_eval = st.button("开始批量评测", type="primary")


def evaluate_model_under_attack(
        defense_name,
        checkpoint_path,
        pre_defense_name,
        dataset,
        max_samples,
):
    model = load_classifier_cached(
        checkpoint_path=checkpoint_path,
        defended_model=defended_model,
        dataset=dataset_name,
        normal=normal,
        pre_defense_name=pre_defense_name,
        gpu=gpu,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    total = 0
    clean_correct = 0
    adv_correct = 0
    attack_success = 0

    progress = st.progress(0)
    status = st.empty()

    processed = 0

    for batch_idx, batch in enumerate(loader):
        points, target = batch

        # target shape: [B, 1] -> [B]
        if target.ndim > 1:
            target = target[:, 0]

        points = points[:, :, :3].float().to(device)  # [B, N, 3]
        target = target.long().to(device)

        remain = max_samples - processed
        if remain <= 0:
            break

        if points.shape[0] > remain:
            points = points[:remain]
            target = target[:remain]

        B = points.shape[0]

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
                cw_lr=3e-3,
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

        total += B
        processed += B

        progress.progress(min(processed / max_samples, 1.0))
        status.text(f"{defense_name}: 已评测 {processed}/{max_samples}")

        if processed >= max_samples:
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
        st.warning("请至少选择一个防御模型。")
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
                    dataset=dataset,
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
            use_container_width=True,
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
# 10. 方法说明
# ============================================================

with st.expander("📘 系统中包含的攻击方法说明"):
    st.markdown(
        """
| 攻击方法 | 类型 | 说明 |
|---|---|---|
| None | 无攻击 | 直接测试 clean accuracy |
| Jitter | 随机噪声攻击 | 对每个点加入高斯噪声并 clip |
| Drop | 点丢弃攻击 | 随机删除部分点 |
| FGM | 梯度攻击 | 单步 Fast Gradient Method |
| IFGM | 迭代梯度攻击 | Iterative FGM |
| MIFGM | 动量迭代攻击 | Momentum Iterative FGM |
| PGD | 投影梯度攻击 | 随机初始化后进行投影梯度攻击 |
| C&W | 优化攻击 | 基于优化的 C&W point perturbation attack |
"""
    )

with st.expander("📘 系统中包含的防御方法说明"):
    st.markdown(
        """
| 防御方法 | 类型 | 说明 |
|---|---|---|
| PointCAT | 训练型防御 | 原 baseline，Contrastive Adversarial Training |
| MA | 训练型防御 | Multi-Attack training，训练时 attack pool 包含 generator / fgm / jitter 等 |
| MA+CC | 训练型防御 | Multi-Attack + Cross-Attack Consistency Loss |
| SOR | 预处理防御 | Statistical Outlier Removal，删除异常点 |
| SRS | 预处理防御 | Simple Random Sampling，随机丢点 |
| DUP-Net | 预处理防御 | SOR + PU-Net upsampling |
"""
    )
