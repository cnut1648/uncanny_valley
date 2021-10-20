from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from torch import nn
import torch
from datasets import Dataset, load_from_disk, concatenate_datasets, DatasetDict
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

# Simple MLP
def MLP_factory(
        layer_sizes,
        dropout=False,
        layer_norm=False,
        activation='gelu'):
    activation_classes = {'gelu': nn.GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}
    modules = nn.ModuleList()
    unpacked_sizes = []
    for block in layer_sizes:
        unpacked_sizes.extend([block[0]] * block[1])

    for k in range(len(unpacked_sizes) - 1):
        if layer_norm:
            modules.append(nn.LayerNorm(unpacked_sizes[k]))

        modules.append(nn.Linear(unpacked_sizes[k], unpacked_sizes[k + 1]))

        if k < len(unpacked_sizes) - 2:
            modules.append(activation_classes[activation.lower()]())
            if dropout is not False:
                modules.append(nn.Dropout(dropout))

    mlp = nn.Sequential(*modules)
    return mlp

class MLP(nn.Module):
  def __init__(self, **kwargs):
    super(MLP, self).__init__()
    self.mlp = MLP_factory(**kwargs)
  def forward(self, X):
    return self.mlp(X)

# find train+test split from this
DATA_PATH = "/home/jiashu/uncanny_valley/datasets/motif/roberta-large/filter-0_test-0.2"
def get_df_from_ds(ds, df):
    texts = ds['text']
    return df.query('text in @texts')

# emb_str = "roberta"
# emb_str = "LF"
emb_str = "SBERT"

DATASET_DIR = "/home/jiashu/uncanny_valley/datasets/ATU.h5"
info = {
    "roberta": {
        "hidden_dim": 1024,
        "name": "roberta-large"
    },
    "SBERT": {
        "name": "SBERT-v2",
        "hidden_dim": 768,
    },
    "LF": {
        "name": "LF-base-4096",
        "hidden_dim": 768,
    }
}

EPOCH = 100
def train(model):
    train_df = get_df_from_ds(train_ds, df)
    test_df = get_df_from_ds(test_ds, df)
    name = info[emb_str]['name']
    ds = TensorDataset(
        torch.tensor(train_df[name].tolist()).to(device),
        torch.tensor(train_df["label"].tolist()).to(device))
    train_dataloader = DataLoader(ds, 
        batch_size=64)
    ds = TensorDataset(
        torch.tensor(test_df[name].tolist()).to(device),
        torch.tensor(test_df["label"].tolist()).to(device))
    test_dataloader = DataLoader(ds, 
        batch_size=64)

    dataloaders = {
        "train": train_dataloader,
        "test": test_dataloader
    }

    for epoch in range(EPOCH):
        probs = []
        labels = []
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for X, y in dataloaders[phase]:
                optim.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(X)
                    loss = criterion(logits, y)
                    if phase == 'train':
                        loss.backward()
                        optim.step()
                    probs.extend(
                        logits.softmax(-1).detach().cpu().numpy()
                    )
                    labels.extend(
                        y.detach().cpu().numpy()
                    )
        if epoch % 5 == 0:
            probs = np.array(probs)
            labels = np.array(labels)
            preds = probs.argmax(-1)
            assert phase == "test", probs.shape[0] == len(test_df)
            macro_auc = roc_auc_score(labels, probs, average="macro", multi_class="ovo")
            weight_auc = roc_auc_score(labels, probs, average="weighted", multi_class="ovo")
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
            print(f"{epoch} test acc => {accuracy_score(preds, labels)}\n"
                f"\t f1 => {f1}\n"
                f"\t precision => {precision}\n"
                f"\t recall => {recall}\n"
                f"\t macro_auc => {macro_auc}\n"
                f"\t weighted auc => {weight_auc}\n"
            )


if __name__ == '__main__':
    df = pd.read_hdf(
        DATASET_DIR, key="default"
    )
    dataset = load_from_disk(DATA_PATH)
    train_ds, test_ds = dataset["train"], dataset['test']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MLP(hidden_dim=info[emb_str]['hidden_dim']).to(device)
    model = MLP(layer_sizes=[
        [info[emb_str]['hidden_dim'], 1],
        [info[emb_str]['hidden_dim'] * 2, 2],
        [7, 1]
    ], dropout=0.3, layer_norm=True, activation='relu').to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optim = optim.Adam(model.parameters(), lr=1e-5)
    train(model)



# def get_model():
#   clf = keras.Sequential([
#       keras.layers.Dense(768, input_shape = (768,), activation = 'relu'),
#       keras.layers.Dropout(0.1),
#       keras.layers.Dense(num_classes, activation = 'softmax')
#   ])
#   clf.compile(
#       loss = 'categorical_crossentropy',
#       optimizer = "adam",
#       metrics=['accuracy']
#   )
#   return clf

# clf = KerasClassifier(
#     build_fn=get_model, 
#     epochs=200,
#     batch_size=32,
#     verbose=5
# )
# kfold = KFold(n_splits=5, shuffle=True)
# results = cross_val_score(clf, X, y, cv=kfold)
# print(results)