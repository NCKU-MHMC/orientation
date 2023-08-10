# 新生訓練 Deep Learning 作業 1

## 作業說明

**目標：訓練 [ImageNet 32x32](https://patrykchrabaszcz.github.io/Imagenet32/) 的分類器**

[ImageNet 32x32](https://patrykchrabaszcz.github.io/Imagenet32/) 原先就分成 training set 與 validation set，其中 training set 又再拆分出 train_data_batch_1 到 train_data_batch_10 個 chunk。

在本次作業中，請將 validation set 當成測試資料。並且在訓練模型時**不可使用 train_data_batch_3 到 train_data_batch_10 中的人工標記**。

## 範例程式執行
### 環境設置
此範例程式依賴下列套件，除了使用 pip 手動安裝外，也推薦使用 [Poetry](./#Poetry) 建制環境。
- torch
- numpy
- hydra-core
- einops
- tqdm
- tensorboard
- jaxtyping

#### Poetry
> 請先安裝 python 3.10 以上版本與 [poetry](https://python-poetry.org/docs/)，建議可使用 [pyenv](https://github.com/pyenv/pyenv#automatic-installer) 管理 python 版本。
```sh
cd <範例程式目錄>
poetry install # 安裝依賴套件
```

### 執行訓練
在 `conf` 資料夾中已經預設 default.yaml 與 l2_regul.yaml 兩組訓練配置，可依據需求自行添加新的設置檔。

```sh
cd <範例程式目錄>
poetry shell # 啟動 virtual environment
python ./main.py --config-name=<config_file_name> # e.g., --config-name=l2_regul 
```

若要暫時調整部份訓練設置請參閱 [hydra docs](https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/)。
