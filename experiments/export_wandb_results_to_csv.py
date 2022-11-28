#!/usr/bin/env python3
import wandb
from omegaconf import OmegaConf


def export_to_csv(run_id, name: str):
    api = wandb.Api()

    # run is specified by <entity>/<project>/<run_id>
    run = api.run(run_id)

    # save the metrics for the run to a csv file
    metrics_dataframe = run.history()
    metrics_dataframe.to_csv("./saved_runs/csv/" + name + "-metrics.csv")


saved_runs = OmegaConf.load("./saved_runs.yaml")
export_to_csv(saved_runs.joint_gating.id, name="joint-gating")
