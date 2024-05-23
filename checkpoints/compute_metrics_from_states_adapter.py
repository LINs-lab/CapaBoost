import argparse
import json
import numpy as np
import logging

SEEDS = [42, 43]

def cola(paths: list, epoch: int) -> None:
    test_matthews_correlations = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        for i in range(len(data["log_history"])):
            test_dict = data["log_history"][i]
            if int(test_dict["epoch"]) == epoch and "test_matthews_correlation" in test_dict.keys():
                test_matthews_correlations.append(test_dict["test_matthews_correlation"]*100)
    mean = np.mean(test_matthews_correlations)
    std = np.std(test_matthews_correlations)
    logging.info(f"seed1: {test_matthews_correlations[0]}, seed2: {test_matthews_correlations[1]}, seed3: {test_matthews_correlations[2]}")
    logging.info(f"Mean: {mean}, std: {std}")

def rte(paths: list, epoch: int) -> None:
    test_accuracy = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        for i in range(len(data["log_history"])):
            test_dict = data["log_history"][i]
            if int(test_dict["epoch"]) == epoch and "test_accuracy" in test_dict.keys():
                test_accuracy.append(test_dict["test_accuracy"]*100)
    mean = np.mean(test_accuracy)
    std = np.std(test_accuracy)
    logging.info(f"seed1: {test_accuracy[0]}, seed2: {test_accuracy[1]}, seed3: {test_accuracy[2]}")
    logging.info(f"Mean: {mean}, std: {std}")

def mrpc(paths: list, epoch: int) -> None:
    test_accuracy = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        for i in range(len(data["log_history"])):
            test_dict = data["log_history"][i]
            if int(test_dict["epoch"]) == epoch and "test_accuracy" in test_dict.keys():
                test_accuracy.append(test_dict["test_accuracy"]*100)
    mean_accuracy = np.mean(test_accuracy)
    std_accuracy = np.std(test_accuracy)
    logging.info(f"seed1-acc: {test_accuracy[0]}, seed2-acc: {test_accuracy[1]}, seed3-acc: {test_accuracy[2]}")
    logging.info(f"Mean accuracy: {mean_accuracy}, std accuracy: {std_accuracy}")

def mnli(paths: list, epoch: int) -> None:
    test_accuracy = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        for i in range(len(data["log_history"])):
            test_dict = data["log_history"][i]
            if int(test_dict["epoch"]) == epoch and "test_accuracy" in test_dict.keys():
                test_accuracy.append(test_dict["test_accuracy"]*100)
    mean = np.mean(test_accuracy)
    std = np.std(test_accuracy)
    logging.info(f"seed1: {test_accuracy[0]}, seed2: {test_accuracy[1]}, seed3: {test_accuracy[2]}")
    logging.info(f"Mean: {mean}, std: {std}")

def stsb(paths: list, epoch: int) -> None:
    test_avg_corr = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        for i in range(len(data["log_history"])):
            test_dict = data["log_history"][i]
            if int(test_dict["epoch"]) == epoch and "test_combined_score" in test_dict.keys():
                test_avg_corr.append(test_dict["test_combined_score"]*100)
    mean_corr = np.mean(test_avg_corr)
    std_corr = np.std(test_avg_corr)
    logging.info(f"seed1-avg_corr: {test_avg_corr[0]}, seed2-avg_corr: {test_avg_corr[1]}, seed3-avg_corr: {test_avg_corr[2]}")
    logging.info(f"Mean avg_corr: {mean_corr}, std avg_corr: {std_corr}")

def qqp(paths: list, epoch: int) -> None:
    test_accuracy = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        for i in range(len(data["log_history"])):
            test_dict = data["log_history"][i]
            if int(test_dict["epoch"]) == epoch and "test_accuracy" in test_dict.keys():
                test_accuracy.append(test_dict["test_accuracy"]*100)
    mean_accuracy = np.mean(test_accuracy)
    std_accuracy = np.std(test_accuracy)
    logging.info(f"seed1-acc: {test_accuracy[0]}, seed2-acc: {test_accuracy[1]}, seed3-acc: {test_accuracy[2]}")
    logging.info(f"Mean accuracy: {mean_accuracy}, std accuracy: {std_accuracy}")

def sst2(paths: list, epoch: int) -> None:
    test_accuracy = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        for i in range(len(data["log_history"])):
            test_dict = data["log_history"][i]
            if int(test_dict["epoch"]) == epoch and "test_accuracy" in test_dict.keys():
                test_accuracy.append(test_dict["test_accuracy"]*100)
    mean_accuracy = np.mean(test_accuracy)
    std_accuracy = np.std(test_accuracy)
    logging.info(f"seed1-acc: {test_accuracy[0]}, seed2-acc: {test_accuracy[1]}, seed3-acc: {test_accuracy[2]}")
    logging.info(f"Mean accuracy: {mean_accuracy}, std accuracy: {std_accuracy}")

def qnli(paths: list, epoch: int) -> None:
    test_accuracy = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        for i in range(len(data["log_history"])):
            test_dict = data["log_history"][i]
            if int(test_dict["epoch"]) == epoch and "test_accuracy" in test_dict.keys():
                test_accuracy.append(test_dict["test_accuracy"]*100)
    mean_accuracy = np.mean(test_accuracy)
    std_accuracy = np.std(test_accuracy)
    logging.info(f"seed1-acc: {test_accuracy[0]}, seed2-acc: {test_accuracy[1]}, seed3-acc: {test_accuracy[2]}")
    logging.info(f"Mean accuracy: {mean_accuracy}, std accuracy: {std_accuracy}")

def main(args):
    paths = []
    for seed in SEEDS:
        # paths.append(f"{args.dataset}/weight-tied/multi-mask_weight-tied.sd_{seed}.lora_r_32.lora_alpha_48.layer_num_{args.num_layer}.density_{args.density}.lr_{args.lr}.epoch_{args.num_epoch}.specifc_epoch/test_results.json")
        # paths.append(f"{args.dataset}/weight-tied_adapter/multi-mask.weight-tied.adapter.sd_{seed}.arf_12.num_layer_{args.num_layer}.density_{args.density}.lr_{args.lr}.num_epoch_{args.num_epoch}.specifc_epoch/test_results.json")
        # paths.append(f"{args.dataset}/parallel-weight-tied_adapter/multi-mask.weight-tied.adapter.sd_{seed}.arf_12.num_layer_{args.num_layer}.density_{args.density}.lr_{args.lr}.num_epoch_{args.num_epoch}.specifc_epoch/test_results.json")
        paths.append(f"{args.dataset}/deberta/parallel-weight-tied_adapter/multi-mask.weight-tied.adapter.sd_{seed}.arf_{args.arf}.num_layer_{args.num_layer}.density_{args.density}.lr_{args.lr}.num_epoch_30.specifc_epoch/trainer_state.json")
    if args.dataset == "cola":
        cola(paths, epoch=args.num_epoch)
    elif args.dataset == "mrpc":
        mrpc(paths, epoch=args.num_epoch)
    elif args.dataset == "rte":
        rte(paths, epoch=args.num_epoch)
    elif args.dataset == "stsb":
        stsb(paths, epoch=args.num_epoch)
    elif args.dataset == "mnli":
        mnli(paths, epoch=args.num_epoch)
    elif args.dataset == "qqp":
        qqp(paths, epoch=args.num_epoch)
    elif args.dataset == "sst2":
        sst2(paths, epoch=args.num_epoch)
    elif args.dataset == "qnli":
        qnli(paths, epoch=args.num_epoch)
    else:
        raise ValueError(f"Dataset {args.dataset} is not supported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lr", type=str, required=True)
    parser.add_argument("--num-epoch", type=int, required=True)
    parser.add_argument("--num-layer", type=int, required=True)
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--arf", type=int, default=12)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(args)