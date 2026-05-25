import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from model_wrapper import LlamaWrapper, QwenWrapper
from dataset import UniDataset
from get_results import get_raw_BASE_results, get_raw_results
from get_vec import uni_generate_vectors
from get_score import get_score
from strategy import UniStrategy
from globalenv import MODEL, LAYERS, Anth_MAIN, Anth_NAME_MAIN

METHOD = "md"
ACTS_PRE = "standard"

ANALYSIS_DIR = "./Analysis"
FIG_DIR = "./figures"
SCORE_ROOT = "./Score-standard"

os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def build_model():
    print(f"[Model] Loading model: {MODEL}")

    if "Llama" in MODEL:
        return LlamaWrapper(MODEL)
    elif "Qwen" in MODEL:
        return QwenWrapper(MODEL)
    else:
        raise NotImplementedError(f"Unsupported model: {MODEL}")

def task_display_name(task):
    return Anth_NAME_MAIN.get(task, task)

def run_all_experiments(model, force=False):
    """
    跑完整实验：
    1. base result
    2. generate steering vectors
    3. calculate LayerNavigator score
    4. run Top-1 / Top-3 / Top-5 steering
    5. save summary.csv / summary.json
    """

    summary_path = os.path.join(ANALYSIS_DIR, "summary.csv")

    if os.path.exists(summary_path) and not force:
        print(f"[Skip] Found existing {summary_path}")
        print("[Skip] Use --force if you want to rerun all experiments.")
        return pd.read_csv(summary_path)

    records = []

    for task in Anth_MAIN:
        print("\n" + "=" * 80)
        print(f"[Task] {task} | {task_display_name(task)}")
        print("=" * 80)

        # 1. test dataset
        test_dataset = UniDataset(
            task=task,
            train=False,
            set="test",
        )

        # 2. base result
        print("[Run] Base result")
        base_prob = get_raw_BASE_results(
            model=model,
            test_dataset=test_dataset,
        )

        print(f"[Result] Base Prob = {base_prob}")

        records.append({
            "task": task,
            "task_name": task_display_name(task),
            "method": "base",
            "num_layers": 0,
            "layers": "",
            "prob": float(base_prob),
        })

        # 3. train dataset
        train_dataset = UniDataset(
            task=task,
            train=True,
            set="train",
        )

        # 4. generate vectors
        print("[Run] Generate steering vectors")
        uni_generate_vectors(
            method=METHOD,
            model=model,
            layers=LAYERS,
            dataset=train_dataset,
        )

        # 5. calculate score
        print("[Run] Calculate LayerNavigator score")
        get_score(
            layers=LAYERS,
            dataset=train_dataset,
            vec_task=task,
            vec_method=METHOD,
            acts_pre=ACTS_PRE,
        )

        # 6. steering results
        for num_layers in [1, 3, 5]:
            print(f"[Run] LayerNavigator Top-{num_layers}")

            strategy = UniStrategy(
                task=task,
                strategy="my",
                num_layers=num_layers,
                method=METHOD,
            )

            test_prob = get_raw_results(
                model=model,
                layers=strategy.layers,
                test_dataset=test_dataset,
                Alphas=[1.0] * num_layers,
                train_task=task,
                train_method=METHOD,
            )

            print(
                f"[Result] Top-{num_layers} | "
                f"Layers = {strategy.layers} | "
                f"Prob = {test_prob}"
            )

            records.append({
                "task": task,
                "task_name": task_display_name(task),
                "method": "LayerNavigator",
                "num_layers": num_layers,
                "layers": ",".join(map(str, strategy.layers)),
                "prob": float(test_prob),
            })

        del train_dataset
        del test_dataset

    df = pd.DataFrame(records)

    csv_path = os.path.join(ANALYSIS_DIR, "summary.csv")
    json_path = os.path.join(ANALYSIS_DIR, "summary.json")

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

    print(f"\n[Saved] {csv_path}")
    print(f"[Saved] {json_path}")

    return df

def load_layer_scores(task):
    """
    读取 Score-standard/{task}/{task}+md/L*.json
    """

    score_dir = os.path.join(SCORE_ROOT, task, f"{task}+{METHOD}")

    if not os.path.exists(score_dir):
        raise FileNotFoundError(f"Score directory not found: {score_dir}")

    rows = []

    for fname in os.listdir(score_dir):
        if not fname.startswith("L") or not fname.endswith(".json"):
            continue

        layer = int(fname[1:-5])
        path = os.path.join(score_dir, fname)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows.append({
            "task": task,
            "task_name": task_display_name(task),
            "layer": layer,
            "s_score": float(data["s_score"]),
            "d_score": float(data["d_score"]),
            "c_score": float(data["c_score"]),
        })

    df = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    return df

def plot_score_curve_for_each_task():
    """
    每个任务单独画一张：
    s_score / d_score / c_score 曲线
    """

    print("\n[Plot] Score curves for each task")

    for task in Anth_MAIN:
        df = load_layer_scores(task)

        top5_layers = (
            df.sort_values("s_score", ascending=False)
            .head(5)["layer"]
            .tolist()
        )

        plt.figure(figsize=(9, 5))

        plt.plot(df["layer"], df["s_score"], marker="o", label="S score = D + C")
        plt.plot(df["layer"], df["d_score"], marker="s", label="Discriminability")
        plt.plot(df["layer"], df["c_score"], marker="^", label="Consistency")

        for layer in top5_layers:
            plt.axvline(layer, linestyle="--", alpha=0.35)

        plt.title(f"{task_display_name(task)}: LayerNavigator Score")
        plt.xlabel("Layer")
        plt.ylabel("Score")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(FIG_DIR, f"{task}_score_curve.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[Saved] {save_path}")
        print(f"[Top-5] {task}: {top5_layers}")

def plot_all_tasks_s_score():
    """
    六个任务的 s_score 画在一张图里
    """

    print("\n[Plot] All tasks S-score curve")

    all_df = pd.concat(
        [load_layer_scores(task) for task in Anth_MAIN],
        ignore_index=True,
    )

    plt.figure(figsize=(10, 6))

    for task in Anth_MAIN:
        sub = all_df[all_df["task"] == task].sort_values("layer")

        plt.plot(
            sub["layer"],
            sub["s_score"],
            marker="o",
            label=task_display_name(task),
        )

    plt.title("LayerNavigator S-score Across Tasks")
    plt.xlabel("Layer")
    plt.ylabel("S-score")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "all_tasks_s_score.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")

def plot_selected_layers_heatmap(topk=5):
    """
    画 Top-K 被选层 heatmap
    """

    print(f"\n[Plot] Selected Top-{topk} layers heatmap")

    first_df = load_layer_scores(Anth_MAIN[0])
    layers = first_df["layer"].tolist()
    num_layers = len(layers)

    mat = np.zeros((len(Anth_MAIN), num_layers))

    rows_for_csv = []

    for i, task in enumerate(Anth_MAIN):
        df = load_layer_scores(task)

        top_df = (
            df.sort_values("s_score", ascending=False)
            .head(topk)
            .copy()
        )

        top_layers = top_df["layer"].tolist()

        for layer in top_layers:
            mat[i, layer] = 1

        for rank, row in enumerate(top_df.itertuples(index=False), start=1):
            rows_for_csv.append({
                "task": task,
                "task_name": task_display_name(task),
                "rank": rank,
                "layer": row.layer,
                "s_score": row.s_score,
                "d_score": row.d_score,
                "c_score": row.c_score,
            })

    plt.figure(figsize=(11, 4.5))
    plt.imshow(mat, aspect="auto")

    plt.yticks(range(len(Anth_MAIN)), [task_display_name(t) for t in Anth_MAIN])
    plt.xticks(range(num_layers), range(num_layers))
    plt.xlabel("Layer")
    plt.title(f"Selected Top-{topk} Intervention Layers")
    plt.colorbar(label="Selected")
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, f"selected_top{topk}_layers_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")

    topk_df = pd.DataFrame(rows_for_csv)
    topk_csv = os.path.join(ANALYSIS_DIR, f"top{topk}_layers.csv")
    topk_df.to_csv(topk_csv, index=False)
    print(f"[Saved] {topk_csv}")

def plot_summary_bar(summary_df):
    """
    Base / Top-1 / Top-3 / Top-5 目标概率柱状图
    """

    print("\n[Plot] Summary probability bar chart")

    df = summary_df.copy()

    df["setting"] = df["num_layers"].apply(
        lambda x: "Base" if int(x) == 0 else f"Top-{int(x)}"
    )

    pivot = df.pivot(index="task_name", columns="setting", values="prob")

    order = ["Base", "Top-1", "Top-3", "Top-5"]
    available_order = [x for x in order if x in pivot.columns]
    pivot = pivot[available_order]

    ax = pivot.plot(kind="bar", figsize=(12, 6))

    plt.title("LayerNavigator Steering Performance")
    plt.xlabel("Task")
    plt.ylabel("Target Answer Probability")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "summary_bar.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")

def plot_improvement_over_base(summary_df):
    """
    相比 Base 的提升幅度
    """

    print("\n[Plot] Improvement over base")

    df = summary_df.copy()

    base_dict = (
        df[df["num_layers"] == 0]
        .set_index("task")["prob"]
        .to_dict()
    )

    steer_df = df[df["num_layers"] > 0].copy()

    steer_df["improvement"] = steer_df.apply(
        lambda row: row["prob"] - base_dict[row["task"]],
        axis=1,
    )

    steer_df["setting"] = steer_df["num_layers"].apply(
        lambda x: f"Top-{int(x)}"
    )

    pivot = steer_df.pivot(
        index="task_name",
        columns="setting",
        values="improvement",
    )

    order = ["Top-1", "Top-3", "Top-5"]
    available_order = [x for x in order if x in pivot.columns]
    pivot = pivot[available_order]

    pivot.plot(kind="bar", figsize=(12, 6))

    plt.axhline(0, linewidth=1)
    plt.title("Improvement over Base Model")
    plt.xlabel("Task")
    plt.ylabel("Probability Improvement")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "improvement_over_base.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")

def export_all_layer_scores():
    """
    导出所有层的 score 排名，方便写报告
    """

    print("\n[Export] Layer score ranking")

    all_rows = []

    for task in Anth_MAIN:
        df = load_layer_scores(task)
        df = df.sort_values("s_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        all_rows.append(df)

    all_df = pd.concat(all_rows, ignore_index=True)

    out_path = os.path.join(ANALYSIS_DIR, "layer_score_ranking.csv")
    all_df.to_csv(out_path, index=False)

    print(f"[Saved] {out_path}")

def run_all_plots(summary_df):
    plot_score_curve_for_each_task()
    plot_all_tasks_s_score()
    plot_selected_layers_heatmap(topk=1)
    plot_selected_layers_heatmap(topk=3)
    plot_selected_layers_heatmap(topk=5)
    plot_summary_bar(summary_df)
    plot_improvement_over_base(summary_df)
    export_all_layer_scores()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="只画图，不重新跑实验。要求 Analysis/summary.csv 和 Score-standard 已经存在。",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="即使 Analysis/summary.csv 已存在，也重新跑完整实验。",
    )

    args = parser.parse_args()

    if args.plot_only:
        summary_path = os.path.join(ANALYSIS_DIR, "summary.csv")

        if not os.path.exists(summary_path):
            raise FileNotFoundError(
                f"Cannot find {summary_path}. "
                f"Please run without --plot-only first."
            )

        summary_df = pd.read_csv(summary_path)
        run_all_plots(summary_df)
        return

    model = build_model()
    summary_df = run_all_experiments(model=model, force=args.force)
    run_all_plots(summary_df)

    print("\n" + "=" * 80)
    print("[Done] All experiments and plots finished.")
    print(f"[Output] CSV/JSON files: {ANALYSIS_DIR}")
    print(f"[Output] Figures: {FIG_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
