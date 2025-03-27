# SPDX-License-Identifier: Apache-2.0
import itertools
import os
from pathlib import Path
from typing import TYPE_CHECKING

import nltk
import numpy
import pandas as pd
import pytest
import yaml

if TYPE_CHECKING:
    import lm_eval as lm_eval_t

# requires a particular lm-evaluation-harness
# pip install lm_eval==0.4.3
lm_eval: "lm_eval_t" = pytest.importorskip("lm_eval",
                                           reason="lm_eval required")

MAX_MODEL_LEN = 4096
RTOL = 0.040
TEST_DATA_PATH = os.environ.get(
    "LM_EVAL_TEST_DATA_FILE",
    "../neuralmagic/lm-eval-configs/models/Meta-Llama-3-8B-Instruct.yaml")
# just show the test data file from the `neuralmagic/lm-eval-configs/models`
# directory.  this could be a `model.yaml`, or a `leaderboard/model.yaml`
TEST_DATA_FILE = str(Path(TEST_DATA_PATH)).replace(
    str(Path.cwd() / "../neuralmagic/lm-eval-configs/models"), "")


def launch_lm_eval(eval_config, tp_size):
    model_args = {
        "pretrained": eval_config['model_name'],
    }
    eval_config_model_args = eval_config.get('model_args')
    if eval_config_model_args:
        model_args.update(eval_config_model_args)

    model_backend = eval_config.get("backend", "vllm")

    if model_backend == "vllm":
        model_args.update({
            "tensor_parallel_size": tp_size,
            "distributed_executor_backend": "ray",
            "max_model_len": MAX_MODEL_LEN
        })

    evaluate_args = {
        "model": model_backend,
        "model_args": ",".join([f"{k}={v}" for k, v in model_args.items()]),
        "tasks": [task["name"] for task in eval_config["tasks"]],
        "num_fewshot": eval_config["num_fewshot"],
        "batch_size": "auto"
    }
    if "limit" in eval_config:
        evaluate_args["limit"] = eval_config["limit"]
    if "fewshot_as_multiturn" in eval_config:
        evaluate_args["fewshot_as_multiturn"] = eval_config[
            "fewshot_as_multiturn"]
    if "apply_chat_template" in eval_config:
        evaluate_args["apply_chat_template"] = eval_config[
            "apply_chat_template"]

    simple_eval_args = ['{}={}'.format(k, v) for k, v in evaluate_args.items()]
    print(f"lm_eval.simple_evaluate({', '.join(simple_eval_args)}")
    results = lm_eval.simple_evaluate(**evaluate_args)

    return results


# pass the TEST_DATA_FILE in as a parameter so that the results
# are uniquely reported to TestMo
@pytest.mark.parametrize("test_data_file", [TEST_DATA_FILE])
def test_lm_eval_correctness(num_gpus_available, test_data_file):
    eval_config = yaml.safe_load(
        Path(TEST_DATA_PATH).read_text(encoding="utf-8"))
    eval_config_tasks = {
        t['name']: {
            m['name']: m['value']
            for m in t['metrics']
        }
        for t in eval_config["tasks"]
    }
    # identify unique metrics we wish to report on.
    eval_config_metrics = set(
        itertools.chain.from_iterable([
            metric.keys() for metric in
            [eval_config_tasks[task] for task in eval_config_tasks]
        ]))

    # retrieve the ground truth values from the evaluation config
    # we transpose the info into a set of records indexed by
    # a "task" and "metric".  The `dropna()` is necessary to remove extra
    # rows where there is no ground truth value for the "task" and "metric"
    ground_truth_df = pd.DataFrame.from_records(
        eval_config_tasks, index=eval_config_metrics).transpose()
    gt_listing_df = ground_truth_df.reset_index(names="task").melt(
        id_vars="task", var_name="metric",
        value_name="ground_truth").dropna().set_index(["task", "metric"])

    # the ifeval task requires an additional set of data
    if "leaderboard_ifeval" in [task["name"] for task in eval_config["tasks"]]:
        nltk.download('punkt_tab')

    # Launch eval requests.
    results = launch_lm_eval(eval_config, tp_size=num_gpus_available)

    # process the results into a dataframe that looks like the ground truth
    # with records indexed by "task" and "metric", but with the measured value
    # for each index.
    results_df = pd.DataFrame.from_records(
        results["results"], index=eval_config_metrics).transpose()
    r_listing_df = (results_df.reset_index(names="task").melt(
        id_vars="task", var_name="metric",
        value_name="measured").dropna().set_index(["task", "metric"]))

    # present the results
    # combine the ground truth and results into a single dataframe
    # but eliminate any rows that do not have both values
    # (This could happen if the eval_config includes a measure that's not
    # generated, or if the LM Evaluation harness generates a measure that
    # was not requested by the eval_config.)
    comparing_metrics_df = pd.concat(
        [gt_listing_df, r_listing_df],
        axis="columns").reset_index(names=["task", "metric"]).dropna()

    # Add a column with the relative tolerance level for the task
    task_rtol_map = {
        t["name"]: t.get("rtol", RTOL)
        for t in eval_config["tasks"]
    }
    comparing_metrics_df.loc[:, "rtol"] = comparing_metrics_df.apply(
        lambda metric: task_rtol_map[metric.task], axis=1)

    # and determine if measured is close to ground truth
    comparing_metrics_df.loc[:, "isclose"] = comparing_metrics_df.apply(
        lambda metric: numpy.isclose(
            metric.ground_truth, metric.measured, rtol=metric.rtol),
        axis=1)
    print("==== LM EVAL RESULT ====\n")
    comparing_metrics_df.sort_values(by=["task", "metric"], inplace=True)
    print(comparing_metrics_df.to_markdown(index=False))

    # save the results for later summary
    llm_results_md = Path("llmeval_results-" +
                          TEST_DATA_FILE.replace("/", "-")).with_suffix(".md")
    llm_results_md.write_text(
        f"## {eval_config['model_name']}\n"
        f"{comparing_metrics_df.to_markdown(index=False)}\n")

    # fail if any scores fail to match ground truth.
    assert comparing_metrics_df.loc[:, "isclose"].all()
