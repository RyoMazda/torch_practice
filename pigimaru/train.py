import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)


def down_sample(
    df: pd.DataFrame,
    down_sampling_ratio: float,
    random_seed: int,
) -> pd.DataFrame:
    """Down Sample DataFrame with "label" column (pos: label == 1, neg: label == 0)
    so that n_neg == down_sampling_ratio * n_pos
    assuming original n_neg is much larger than n_pos.
    """
    is_pos = df.label == 1
    n_pos = is_pos.sum()
    n_neg = len(df) - n_pos
    if n_neg > n_pos * down_sampling_ratio:
        df_pos = df[is_pos]
        df_neg = df[~is_pos].sample(
            n=int(n_pos * down_sampling_ratio),
            random_state=random_seed,
        )
        return pd.concat([df_pos, df_neg]).reset_index(drop=True).copy()
    else:
        # We don't have to down-sample because the data is well-balanced enough.
        return df


@dataclass
class MyBatch:
    text: torch.LongTensor
    mask: torch.LongTensor
    label: torch.FloatTensor


def get_data_loader(
    texts: List[str],
    labels: np.ndarray,
    tokenizer: transformers.BertJapaneseTokenizer,
    batch_size: int,
    max_length: int = 512,
) -> Iterable[MyBatch]:
    def collate_fn(
        batch: List[torch.Tensor],
    ) -> MyBatch:
        text_tensor = torch.stack([text for text, _, _ in batch])
        mask_tensor = torch.stack([mask for _, mask, _ in batch])
        y_tensor = torch.stack([y for _, _, y in batch]).to(torch.float32)  # type: ignore
        return MyBatch(text=text_tensor, mask=mask_tensor, label=y_tensor)  # type: ignore

    y = torch.tensor(labels, dtype=torch.int64)  # type: ignore
    x = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return torch.utils.data.DataLoader(  # type: ignore
        dataset=torch.utils.data.TensorDataset(x['input_ids'], x['attention_mask'], y),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )


class PrecisionWithRecallGuaranteeScore:
    def __init__(self, recall_guarantee: float = 0.95) -> None:
        self.recall_guarantee = recall_guarantee

    def __call__(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        for th in logits.sort(descending=True)[0]:
            precision, recall, _, _ = precision_recall_fscore_support(
                y.cpu().detach().numpy(),
                (logits >= th).cpu().detach().numpy(),
                zero_division=0,
                average='binary',
            )
            if recall >= self.recall_guarantee:
                return float(precision)
        return 0.0


def eval_model(
        model: torch.nn.Module,  # type: ignore
        loader: Iterable[MyBatch],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scoring_fn: Callable[[torch.Tensor, torch.Tensor], float],
        device: torch.device = torch.device('cpu'),  # type: ignore
) -> Tuple[float, float]:
    model.to(device)
    model.eval()
    losses = []
    outs = []
    ys = []
    for batch in loader:
        x = batch.text.to(device)
        m = batch.mask.to(device)
        y = batch.label.to(device)
        with torch.no_grad():
            out = model(input_ids=x, attention_mask=m)
        loss = loss_fn(out, y)
        losses.append(loss.item())
        outs.append(out)
        ys.append(y)
    avg_loss = sum(losses) / len(losses)
    score = scoring_fn(
        torch.cat(outs, dim=0),  # type: ignore
        torch.cat(ys, dim=0),  # type: ignore
    )
    return avg_loss, score


def _should_stop(scores: List[float], early_stopping_rounds: int) -> bool:
    """
    Example:
    >>> scores = [0, 1, 2, 3, 4, 4, 5]
    >>> len(scores), np.argmax(scores)
    >>> 7, 4
    Then, the stacking_rounds is 7 - 4 - 1 = 2, that means it didn't improve in the recent 2 rounds.
    """
    stacking_rounds: int = len(scores) - np.argmax(scores) - 1
    return stacking_rounds >= early_stopping_rounds


def train_model(
        model: torch.nn.Module,  # type: ignore
        max_epoch: int,
        early_stopping_rounds: int,
        train_loader: Iterable[Any],
        valid_loader: Iterable[Any],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scoring_fn: Callable[[torch.Tensor, torch.Tensor], float],
        optimizer: torch.optim.Optimizer,
        output_dir: Path,
        device: torch.device = torch.device('cpu'),  # type: ignore
) -> int:
    logger.info("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        logger.info(f"  - {var_name}\t{optimizer.state_dict()[var_name]}")
    best_epoch = 0
    best_score = -1e9
    scores = []
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Start training. Models will be saved at '{output_dir.absolute()}'.")
    logger.info('Epoch\tTime[m]\tTrainLoss\tValidLoss\tScore\n')
    for epoch in range(max_epoch):
        # train
        model.train()
        train_losses = []
        # train_scores = []
        for batch in train_loader:
            x = batch.text.to(device)
            m = batch.mask.to(device)
            y = batch.label.to(device)
            optimizer.zero_grad()
            out = model(input_ids=x, attention_mask=m)
            loss = loss_fn(out, y)
            loss.backward()  # type: ignore
            optimizer.step()
            # For Logging
            train_losses.append(loss.item())
        avg_train_loss = sum(train_losses) / len(train_losses)
        # Validation
        avg_valid_loss, score = eval_model(
            model=model,
            loader=valid_loader,
            loss_fn=loss_fn,
            scoring_fn=scoring_fn,
            device=device,
        )
        scores.append(score)
        if score >= best_score:
            # Remove old model
            # file name still has epoch number since this is temporary due to the lack of starage
            (output_dir / f"epoch_{best_epoch}.pt").unlink(missing_ok=True)
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), str(output_dir / f"epoch_{epoch}.pt"))
        # Log
        elapsed_minutes = round((time.time() - start_time) / 60, 2)
        logger.info(
            f'{epoch}'
            f'\t{elapsed_minutes}'
            f'\t{avg_train_loss:.5f}'
            f'\t{avg_valid_loss:.5f}'
            f'\t{score:.5f}'
        )
        # Early Stopping
        if _should_stop(scores, early_stopping_rounds):
            logger.info('Early Stopping!!')
            break
    logger.info(f"Training finished successfully! Best epoch is {best_epoch}")
    return best_epoch


def __main__() -> None:
    """
    Command Args:
      - 1: path for training data tsv file that has text & label columns
      - 2: path for validation data tsv that has text & label columns
      - 3: path for test data tsv that has text & label columns
    """
    import argparse
    import datetime
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    output_dir = Path(f"./outputs/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('args', nargs='*')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    logger.info(args)
    if len(args.args) != 3:
        raise ValueError("Please input the arguments as written in the document or source.")
    debug = bool(args.debug)
    # Logging Config
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        filename=str(output_dir/'train.log'),
        filemode='w',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    if debug:
        logger.debug("Running in debug mode")
    # paths
    train_path = Path(args.args[0])
    valid_path = Path(args.args[1])
    test_path = Path(args.args[2])
    logger.info(train_path)
    logger.info(valid_path)
    logger.info(test_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # type: ignore
    logger.info(f"device is: {device}")
    # Data
    train = pd.read_csv(train_path, sep='\t')
    logger.info(f"train.shape: {train.shape}")
    valid = pd.read_csv(valid_path, sep='\t')
    logger.info(f"valid.shape: {valid.shape}")
    train = down_sample(train, down_sampling_ratio=2.0, random_seed=102)
    valid = down_sample(valid, down_sampling_ratio=2.0, random_seed=102)
    # Tokenizer
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = transformers.BertJapaneseTokenizer.from_pretrained(model_name)
    # Data Loader
    max_length = 64 if debug else 512
    batch_size = 8 if debug else 64
    train_loader = get_data_loader(
        texts=train.text.to_list(),
        labels=train.label.values,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
    )
    valid_loader = get_data_loader(
        texts=valid.text.to_list(),
        labels=valid.label.values,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
    )
    # Model
    from .models import MyBertModel
    bert_model = transformers.BertModel.from_pretrained(model_name)
    model = MyBertModel(output_dim=1, tokenizer=tokenizer, bert_model=bert_model).to(device)
    # Train
    best_epoch = train_model(
        model=model,
        max_epoch=10 if debug else 100,
        early_stopping_rounds=2 if debug else 5,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=torch.nn.BCEWithLogitsLoss(),  # type: ignore
        scoring_fn=PrecisionWithRecallGuaranteeScore(),
        output_dir=output_dir,
        optimizer=torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01),
        device=device,
    )
    # Evaluation by Test data
    model.load_state_dict(torch.load(  # type: ignore
        str(output_dir / f'epoch_{best_epoch}.pt'),
        map_location=device,
    ))
    model.eval()
    test = pd.read_csv(test_path, sep='\t')
    logger.info(f"test.shape: {test.shape}\tnum_pos: {(test.label==1).sum()}\tnum_neg: {(test.label==0).sum()}")
    test['prob'] = test.text.apply(lambda x: model.predict(x, device=device))
    # Following is too specific to the task. Just temporary
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(test.label.values, test.prob.values)
    precision, recall, _ = precision_recall_curve(
        y_true=test.label.values,
        probas_pred=test.prob,
    )
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve | AUC: {auc:.5f}')
    plt.savefig(str(output_dir / 'precision_recall_curve.png'))
    logger.info("reason\tnum_records\tAUC")
    for target_reason in range(10):
        sub = test[(test.reason_label == target_reason) | (test.label == 0)]
        try:
            auc = str(roc_auc_score(sub.label.values, sub.prob.values))
        except ValueError:
            auc = '-'
        logger.info(f"{target_reason}\t{len(sub[sub.label == 1])}\t{auc}")


if __name__ == '__main__':
    __main__()
