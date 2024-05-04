import logging
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from models.models import CannibalModelForClassification, WSLForClassification
from models.relation_transformer.models import RelationTransformer
from models.relation_transformer.probe_features import INRPerLayer
from experiments.utils.utils import (
    count_parameters,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from experiments.data.dataset import INRDataset
from augmentations.augmentations import mixup_augs, name2aug
from experiments.utils.lr_schedule import WarmupLRScheduler

set_logger()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss = 0.0
    correct = 0.0
    total = 0.0
    predicted, gt = [], []
    for batch in loader:
        batch = batch.to(device)
        inputs = (batch.weights, batch.biases)
        out = model(inputs)
        loss += F.cross_entropy(out, batch.label, reduction="sum")
        total += len(batch.label)
        pred = out.argmax(1)
        correct += pred.eq(batch.label).sum()
        predicted.extend(pred.cpu().numpy().tolist())
        gt.extend(batch.label.cpu().numpy().tolist())

    model.train()
    avg_loss = loss / total
    avg_acc = correct / total

    return dict(avg_loss=avg_loss, avg_acc=avg_acc, predicted=predicted, gt=gt)


def main(args, device):
    path = args.data_path

    mixup_augmentation = mixup_augs.get(args.augmentation, None)
    augmentations = name2aug.get(args.augmentation, None)

    # load dataset
    train_set = INRDataset(
        path=path,
        split="train",
        normalize=args.normalize,
        do_augmentation=args.do_augmentation,
        augmentations=[augmentations] if augmentations is not None else augmentations,
        permutation=args.permutation,
        statistics_path=args.statistics_path,
        mixup_augmentation=mixup_augmentation,
        num_classes=args.num_classes,
    )
    val_set = INRDataset(
        path=path,
        split="val",
        normalize=args.normalize,
        statistics_path=args.statistics_path,
        num_classes=args.num_classes,
    )
    test_set = INRDataset(
        path=path,
        split="test",
        normalize=args.normalize,
        statistics_path=args.statistics_path,
        num_classes=args.num_classes,
    )

    # get subset of views
    available_views = list(sorted(set([p.split("/")[-3] for p in train_set.dataset])))
    views_to_use = available_views[: args.n_views]
    logging.info(f"using views: {views_to_use}")

    train_idx = [
        i for i, p in enumerate(train_set.dataset) if p.split("/")[-3] in views_to_use
    ]
    # NOTE: for test we use only the first view
    test_idx = [
        i
        for i, p in enumerate(test_set.dataset)
        if p.split("/")[-3] in available_views[:1]
    ]

    val_idx = [
        i
        for i, p in enumerate(val_set.dataset)
        if p.split("/")[-3] in available_views[:1]
    ]

    train_set = torch.utils.data.Subset(train_set, train_idx)
    test_set = torch.utils.data.Subset(test_set, test_idx)
    val_set = torch.utils.data.Subset(val_set, val_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size // args.grad_accum_steps,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logging.info(
        f"train size {len(train_set)}, "
        f"val size {len(val_set)}, "
        f"test size {len(test_set)}"
    )

    point = train_set.__getitem__(0)
    weight_shapes = tuple(w.shape[:2] for w in point.weights)
    bias_shapes = tuple(b.shape[:1] for b in point.biases)
    layer_layout = [weight_shapes[0][0]] + [b[0] for b in bias_shapes]

    logging.info(f"weight shapes: {weight_shapes}, bias shapes: {bias_shapes}")

    model = {
        "rtransformer": RelationTransformer(
            d_in=1,
            layer_layout=layer_layout,
            inr_model=INRPerLayer(
                in_dim=weight_shapes[0][0],
                n_layers=len(weight_shapes),
                out_channels=weight_shapes[-1][-1],
                up_scale=weight_shapes[0][1] // weight_shapes[0][0],
            ),
            d_out=args.num_classes,
        ).to(device),
        "cannibal": CannibalModelForClassification(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            input_features=1,
            hidden_dim=args.dim_hidden,
            n_hidden=args.n_hidden,
            reduction=args.reduction,
            n_fc_layers=args.n_fc_layers,
            set_layer=args.set_layer,
            n_out_fc=args.n_out_fc,
            dropout_rate=args.do_rate,
            bn=args.add_bn,
            n_classes=args.num_classes,
        ).to(device),
        "wsl": WSLForClassification(
            n_encode_layers=args.n_encode_layers,
            input_hidd_dim=weight_shapes[0][1],
            n_heads=args.n_heads,
            hid_ff=args.hid_ff,
            dropout=0.1,
            n_classes=10
        ).to(device),
    }[args.model]
    model = nn.DataParallel(model)

    logging.info(f"number of parameters: {count_parameters(model)}")

    optimizer = {
        "adam": torch.optim.Adam(
            [
                dict(params=model.parameters(), lr=args.lr),
            ],
            lr=args.lr,
            weight_decay=5e-4,
        ),
        "sgd": torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9
        ),
        "adamw": torch.optim.AdamW(
            params=model.parameters(), lr=args.lr, amsgrad=True, weight_decay=5e-4
        ),
    }[args.optim]

    scheduler = None
    if args.use_lr_schedule:
        scheduler = WarmupLRScheduler(optimizer, warmup_steps=1000)

    epoch_iter = trange(args.n_epochs)

    criterion = nn.CrossEntropyLoss()
    best_val_acc = -1
    best_test_results, best_val_results = None, None
    test_acc, test_loss = -1.0, -1.0
    step = 0
    optimizer.zero_grad()
    for epoch in epoch_iter:
        for i, batch in enumerate(train_loader):
            model.train()

            batch = batch.to(device)
            inputs = (batch.weights, batch.biases)
            out = model(inputs)

            loss = criterion(out, batch.label)
            loss = loss / args.grad_accum_steps
            loss.backward()

            if step % args.grad_accum_steps == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            if args.wandb:
                log = {
                    "train/loss": loss.item(),
                }
                wandb.log(log)

            epoch_iter.set_description(
                f"[{epoch} {i+1}], train loss: {loss.item():.3f}, test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}"
            )

            step += 1

        if (epoch + 1) % args.eval_every == 0:
            val_loss_dict = evaluate(model, val_loader, device)
            test_loss_dict = evaluate(model, test_loader, device)
            val_loss = val_loss_dict["avg_loss"]
            val_acc = val_loss_dict["avg_acc"]
            test_loss = test_loss_dict["avg_loss"]
            test_acc = test_loss_dict["avg_acc"]

            best_val_criteria = val_acc >= best_val_acc

            if best_val_criteria:
                best_val_acc = val_acc
                best_test_results = test_loss_dict
                best_val_results = val_loss_dict

            if args.wandb:
                log = {
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/best_loss": best_val_results["avg_loss"],
                    "val/best_acc": best_val_results["avg_acc"],
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "test/best_loss": best_test_results["avg_loss"],
                    "test/best_acc": best_test_results["avg_acc"],
                    "epoch": epoch,
                }

                wandb.log(log)


if __name__ == "__main__":
    parser = ArgumentParser(
        "Classification - Cannibal trainer",
    )

    # Data params

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for reproducibility",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="path for dataset",
    )

    parser.add_argument(
        "--statistics-path",
        type=str,
        default="dataset/statistics.pth",
        help="path to dataset statistics",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default="dataset/mnist_splits.json",
        help="path to dataset statistics",
    )

    parser.add_argument(
        "--num-classes", type=int, default=10, help="number of label classes"
    )

    parser.add_argument(
        "--n-epochs",
        type=int,
        default=200,
        help="num epochs",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="train batch size",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        help="test / val batch size",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cannibal",
        choices=["cannibal", "rtransformer", "wsl"],
        help="model",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        choices=["adam", "sgd", "adamw"],
        help="optimizer",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="amount of gradient accumulation steps",
    )
    parser.add_argument(
        "--use-lr-schedule",
        type=str2bool,
        required=True,
        help="to use learning rate scheduler",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="num workers")
    parser.add_argument("--n-views", type=int, default=1, help="number of views")
    parser.add_argument(
        "--reduction",
        type=str,
        default="max",
        choices=["mean", "sum", "max", "attn"],
        help="reduction strategy",
    )
    parser.add_argument(
        "--dim-hidden",
        type=int,
        default=16,
        help="dim hidden layers",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=2,
        help="num hidden layers",
    )
    parser.add_argument(
        "--n-fc-layers",
        type=int,
        default=1,
        help="num linear layers at each ff block",
    )
    parser.add_argument(
        "--n-out-fc",
        type=int,
        default=1,
        help="num linear layers at final layer (invariant block)",
    )
    parser.add_argument(
        "--set-layer",
        type=str,
        default="sab",
        choices=["sab", "ds"],
        help="set layer",
    )

    parser.add_argument(
        "--n-encode-layers",
        type=int,
        default=8,
        help="number of transformer encoder layers",
    )

    parser.add_argument(
        "--hid-ff",
        type=int,
        default=1024,
        help="attntion hidd dim",
    )

    parser.add_argument(
        "--n-heads",
        type=int,
        default=8,
        help="number of attention heads",
    )

    parser.add_argument("--eval-every", type=int, default=1, help="eval every in epochs")
    parser.add_argument(
        "--do-augmentation", type=str2bool, default=False, help="use augmentation"
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default=None,
        choices=list(mixup_augs.keys()) + list(name2aug.keys()) + ["no-aug"],
        help="type of augmentation",
    )

    parser.add_argument(
        "--permutation", type=str2bool, default=False, help="use permutations"
    )
    parser.add_argument(
        "--normalize", type=str2bool, default=False, help="normalize data"
    )

    parser.add_argument("--do-rate", type=float, default=0.0, help="dropout rate")
    parser.add_argument(
        "--add-bn", type=str2bool, default=False, help="add batch norm layers"
    )


    # wandb params
    parser.add_argument("--wandb", dest="wandb", action="store_true")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.set_defaults(wandb=True)

    parser.add_argument("--wandb-project", type=str, default="deep-weight-space-aug")
    parser.add_argument("--wandb-entity", type=str, default="dws")

    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    # wandb
    if args.wandb:
        name = (
            f"inr_classification_{args.lr}_hid_dim_{args.dim_hidden}_reduction_{args.reduction}"
            f"_bs_{args.batch_size}_seed_{args.seed}"
        )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=name,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)

    device = get_device()

    main(
        args=args,
        device=device,
    )
