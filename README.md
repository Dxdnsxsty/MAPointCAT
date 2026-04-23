# 环境配置
```shell
conda create -n mapointcat python=3.8 -y

conda activate mapointcat

conda install scikit-learn scipy matplotlib tqdm pyyaml pillow joblib ninja pandas -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# 构建代码
```shell
git clone https://github.com/Dxdnsxsty/MAPointCAT.git

cd MAPointCAT
```

创建软链接（Linux）：

```shell
ln -s /DATASET_PATH ./data/DATASET_PATH
```

# 运行
## 训练
训练（win）：

```powershell
python trainer.py --experiment_dir pn_train `
--data_path ./data/modelnet40_normal_resampled/ `
--dataset ModelNet40 `
--defended_model pointnet_cls `
--eps 0.04 `
--alpha 8. `
--beta 0.5 `
--use_cosine_similarity `
--inner_loop_nums 4 `
--batch_size 64 `
--init_search_iters 500 `
--update_search_iters 10 `
--lr_fp 0.001 `
--use_multi_gpu `
```

训练（Linux）：

```powershell
python trainer.py --experiment_dir pre_alter_pn2 \
--data_path ./data/modelnet40_normal_resampled/ \
--dataset ModelNet40 \
--defended_model pointnet2_cls_msg \
--eps 0.04 \
--alpha 8. \
--beta 0.5 \
--use_cosine_similarity \
--inner_loop_nums 4 \
--batch_size 64 \
--init_search_iters 500 \
--update_search_iters 10 \
--lr_fp 0.001 \
--use_multi_gpu \
```

## 评估
测试（win）：

```powershell
python tester.py `
--data_path ./data/modelnet40_normal_resampled/ `
--dataset ModelNet40 `
--defended_model pointnet_cls `
--batch_size 16 `
--mode test_normal `
--checkpoint_dir ./log/pn_train/checkpoints/latest-cls.pth
```

测试（Linux）：

```powershell
python tester.py \
--data_path ./data/modelnet40_normal_resampled/ \
--dataset ModelNet40 \
--defended_model pointnet_cls \
--batch_size 256 \
--mode test_normal \
--checkpoint_dir ./log/ \
```

