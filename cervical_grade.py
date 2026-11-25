"""Command-line entry point for paired cervical radiograph grading."""

import argparse
import math
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description="用原始 JPG 和对应的分割 PNG 进行颈椎四级分类")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="训练模型并保存本地断点")
    evaluate = commands.add_parser("evaluate", help="使用本地断点评估带标签的 val 或 test")
    predict = commands.add_parser("predict", help="对单张原片或整个配对目录输出四类概率")
    for command in (train, evaluate, predict):
        command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                             help="默认有 CUDA 时使用 CUDA")
        command.add_argument("--workers", type=int, default=0, help="并行读图进程数，默认 0")
        command.add_argument("--batch-size", type=int, default=None if command is train else 2,
                             help="每批原片数，默认 2，续训时沿用断点设置")
    for command in (train, evaluate):
        command.add_argument("--data-root", type=Path, required=True, help="分级数据根目录")
    split = train.add_mutually_exclusive_group()
    split.add_argument("--split-csv", type=Path, help="按患者划分的 image_path,patient_id,split 表")
    split.add_argument("--train-only", action="store_true", help="使用根目录的分级文件夹，仅训练")
    train.add_argument("--output-dir", type=Path, default=Path("runs/cervical"), help="实验输出目录")
    train.add_argument("--epochs", type=int, default=30, help="训练总轮数，包含已完成的轮次，默认 30")
    train.add_argument("--lr", type=float, help="AdamW 学习率，默认 1e-4")
    train.add_argument("--weight-decay", type=float, help="AdamW 权重衰减，默认 1e-4")
    train.add_argument("--seed", type=int, help="随机种子，默认 42")
    train.add_argument("--weight-by-grade", action="store_true", default=None,
                       help="按训练集等级频次加权序数损失，续训沿用断点设置")
    train.add_argument("--resume", action="store_true", help="从输出目录的 last.pt 继续训练")
    evaluate.add_argument("--split-csv", type=Path, help="与训练相同格式的患者分区表")
    evaluate.add_argument("--split", choices=("val", "test"), default="val")
    evaluate.add_argument("--output", type=Path, help="可选，保存各项指标的 .json 文件")
    for command in (evaluate, predict):
        command.add_argument("--checkpoint", type=Path, required=True, help="本方案本地 best.pt 或 last.pt")
    predict.add_argument("--input", type=Path, required=True, help="一个 JPG/JPEG，或含配对图像的目录")
    predict.add_argument("--output", type=Path, required=True, help="逐张预测的 .csv 文件")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if args.command == "train":
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
            parser.error("--lr must be finite and positive")
        if args.weight_decay is not None and (not math.isfinite(args.weight_decay) or args.weight_decay < 0):
            parser.error("--weight-decay must be finite and nonnegative")
        if args.seed is not None and not 0 <= args.seed < 2**32:
            parser.error("--seed must be between 0 and 2^32-1")
    elif args.output is not None:
        suffix = ".json" if args.command == "evaluate" else ".csv"
        if args.output.suffix.lower() != suffix:
            parser.error(f"--output must use a {suffix} file")

    from cervical.engine import evaluate, predict, train

    try:
        {"train": train, "evaluate": evaluate, "predict": predict}[args.command](args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
