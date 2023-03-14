import random, os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def log_selector(selector1, selector2, mask):
    assert selector1.shape[1] == selector2.shape[1], "different selector heads"

    s1 = selector1[0].long().cpu().detach()
    s2 = selector2[0].cpu().detach()
    mask = mask[0].cpu().detach()

    indices = []
    j = 0
    while len(indices) < 50 and j < s1.shape[1]:
        if mask[j]:
            indices.append(j)
        j += np.random.randint(10)

    print("selectors")
    for i in range(len(s2)):
        print(s2[i].item(), end="\t")
        print(s1[i].sum().item(), "selected", end="\t")
        for j in indices:
            print(s1[i, j].item(), end="")
        print("")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_data(name="ecg", batch_size=64):
    df = pd.read_csv("../dataset/ecg/mitbih.csv")
    data_columns = df.columns[:-2].tolist()
    X = torch.FloatTensor(np.array(df[data_columns].astype("float32")))
    y = torch.LongTensor(np.array(df["class"]))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42, stratify=y
    )
    train_dl = DataLoader(
        TensorDataset(X_train, y_train), batch_size, shuffle=True, num_workers=4
    )
    test_dl = DataLoader(
        TensorDataset(X_test, y_test), batch_size, shuffle=False, num_workers=4
    )

    class_weights = (1 - (np.bincount(y_train) / y_train.shape[0])).tolist()
    max_len = X.shape[1]
    return train_dl, test_dl, class_weights, max_len


def my_optim(params, lr, wd, warmup_steps, use_scheduler=False):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad == True, params),
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    if not use_scheduler:
        return optimizer

    def lr_foo(epoch):
        if epoch < warmup_steps:
            # warm up lr
            lr_scale = 0.1 ** (warmup_steps - epoch)
        else:
            lr_scale = 0.95 ** epoch
        return lr_scale

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)

    return [optimizer], [scheduler]
