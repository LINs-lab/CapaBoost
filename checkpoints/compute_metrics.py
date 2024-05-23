import argparse
import json
import numpy as np
import logging

SEEDS = [42, 43, 44]

def cola(paths: list) -> None:
    eval_matthews_correlations = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        eval_matthews_correlations.append(data["eval_matthews_correlation"]*100)
    mean = np.mean(eval_matthews_correlations)
    std = np.std(eval_matthews_correlations)
    logging.info(f"seed1: {eval_matthews_correlations[0]}, seed2: {eval_matthews_correlations[1]}, seed3: {eval_matthews_correlations[2]}")
    logging.info(f"Mean: {mean}, std: {std}")

def rte(paths: list) -> None:
    eval_accuracy = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        eval_accuracy.append(data["eval_accuracy"]*100)
    mean = np.mean(eval_accuracy)
    std = np.std(eval_accuracy)
    logging.info(f"seed1: {eval_accuracy[0]}, seed2: {eval_accuracy[1]}, seed3: {eval_accuracy[2]}")
    logging.info(f"Mean: {mean}, std: {std}")

def mrpc(paths: list) -> None:
    eval_accuracy = []
    eval_f1 = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        eval_accuracy.append(data["eval_accuracy"]*100)
        eval_f1.append(data["eval_f1"]*100)
    mean_accuracy = np.mean(eval_accuracy)
    std_accuracy = np.std(eval_accuracy)
    mean_f1 = np.mean(eval_f1)
    std_f1 = np.std(eval_f1)
    logging.info(f"seed1-acc: {eval_accuracy[0]}, seed2-acc: {eval_accuracy[1]}, seed3-acc: {eval_accuracy[2]}")
    logging.info(f"seed1-f1: {eval_f1[0]}, seed2-f1: {eval_f1[1]}, seed3-f1: {eval_f1[2]}")
    logging.info(f"Mean accuracy: {mean_accuracy}, std accuracy: {std_accuracy}")
    logging.info(f"Mean f1: {mean_f1}, std f1: {std_f1}")

def mnli(paths: list) -> None:
    eval_accuracy = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        eval_accuracy.append(data["eval_accuracy"]*100)
    mean = np.mean(eval_accuracy)
    std = np.std(eval_accuracy)
    logging.info(f"seed1: {eval_accuracy[0]}, seed2: {eval_accuracy[1]}, seed3: {eval_accuracy[2]}")
    logging.info(f"Mean: {mean}, std: {std}")

def stsb(paths: list) -> None:
    eval_pearson = []
    eval_spearmanr = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        eval_pearson.append(data["eval_pearson"]*100)
        eval_spearmanr.append(data["eval_spearmanr"]*100)
    mean_pearson = np.mean(eval_pearson)
    std_pearson = np.std(eval_pearson)
    mean_spearmanr = np.mean(eval_spearmanr)
    std_spearmanr = np.std(eval_spearmanr)
    logging.info(f"seed1-pearson: {eval_pearson[0]}, seed2-pearson: {eval_pearson[1]}, seed3-pearson: {eval_pearson[2]}")
    logging.info(f"seed1-spearmanr: {eval_spearmanr[0]}, seed2-spearmanr: {eval_spearmanr[1]}, seed3-spearmanr: {eval_spearmanr[2]}")
    logging.info(f"Mean pearson: {mean_pearson}, std pearson: {std_pearson}")
    logging.info(f"Mean spearmanr: {mean_spearmanr}, std spearmanr: {std_spearmanr}")

def qqp(paths: list) -> None:
    eval_accuracy = []
    eval_f1 = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        eval_accuracy.append(data["eval_accuracy"]*100)
        eval_f1.append(data["eval_f1"]*100)
    mean_accuracy = np.mean(eval_accuracy)
    std_accuracy = np.std(eval_accuracy)
    mean_f1 = np.mean(eval_f1)
    std_f1 = np.std(eval_f1)
    logging.info(f"seed1-acc: {eval_accuracy[0]}, seed2-acc: {eval_accuracy[1]}, seed3-acc: {eval_accuracy[2]}")
    logging.info(f"seed1-f1: {eval_f1[0]}, seed2-f1: {eval_f1[1]}, seed3-f1: {eval_f1[2]}")
    logging.info(f"Mean accuracy: {mean_accuracy}, std accuracy: {std_accuracy}")
    logging.info(f"Mean f1: {mean_f1}, std f1: {std_f1}")

def sst2(paths: list) -> None:
    eval_accuracy = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        eval_accuracy.append(data["eval_accuracy"]*100)
    mean = np.mean(eval_accuracy)
    std = np.std(eval_accuracy)
    logging.info(f"seed1: {eval_accuracy[0]}, seed2: {eval_accuracy[1]}, seed3: {eval_accuracy[2]}")
    logging.info(f"Mean: {mean}, std: {std}")

def qnli(paths: list) -> None:
    eval_accuracy = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        eval_accuracy.append(data["eval_accuracy"]*100)
    mean = np.mean(eval_accuracy)
    std = np.std(eval_accuracy)
    logging.info(f"seed1: {eval_accuracy[0]}, seed2: {eval_accuracy[1]}, seed3: {eval_accuracy[2]}")
    logging.info(f"Mean: {mean}, std: {std}")

def main(args):
    paths = []
    for seed in SEEDS:
        paths.append(f"{args.dataset}/weight-tied/multi-mask_weight-tied.sd_{seed}.lora_r_32.lora_alpha_48.layer_num_{args.num_layer}.density_{args.density}.lr_{args.lr}.epoch_{args.num_epoch}.specifc_epoch/test_results.json")
    if args.dataset == "cola":
        cola(paths)
    elif args.dataset == "mrpc":
        mrpc(paths)
    elif args.dataset == "rte":
        rte(paths)
    elif args.dataset == "stsb":
        stsb(paths)
    elif args.dataset == "mnli":
        mnli(paths)
    elif args.dataset == "qqp":
        qqp(paths)
    elif args.dataset == "sst2":
        sst2(paths)
    elif args.dataset == "qnli":
        qnli(paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lr", type=str, required=True)
    parser.add_argument("--num-epoch", type=int, required=True)
    parser.add_argument("--num-layer", type=int, required=True)
    parser.add_argument("--density", type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(args)