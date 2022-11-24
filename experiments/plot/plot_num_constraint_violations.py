#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from omegaconf import OmegaConf


name = "./saved_runs/csv/aidanscannell:mode_constrained_mbrl:12tuiu9j:num_episodes_with_constraint_violations.csv"
df = pd.read_csv(name)
num_violations = df[
    "joint_gating/Pr=0.8/beta=10.0 - Num episodes with constraint violations"
]
name = (
    "./saved_runs/csv/aidanscannell:mode_constrained_mbrl:12tuiu9j:extrinsic_return.csv"
)
df = pd.read_csv(name)
extrinsic_return = df["joint_gating/Pr=0.8/beta=10.0 - Extrinsic return"]
print("num_violations_jiont")
print(num_violations[79])

name = "./saved_runs/csv/aidanscannell:mode_constrained_mbrl:wx5raxbs:num_constraint_violations.csv"
df = pd.read_csv(name)
num_violations_independent = df[
    "independent_gating/Pr=0.8/beta=10.0 - Num episodes with constraint violations"
]
print("num_violations_independent independent")
print(num_violations_independent[79])

name = (
    "./saved_runs/csv/aidanscannell:mode_constrained_mbrl:wx5raxbs:extrinsic_return.csv"
)
df = pd.read_csv(name)
extrinsic_return_independent = df[
    "independent_gating/Pr=0.8/beta=10.0 - Extrinsic return"
]

fig = plt.figure()
gs = fig.add_gridspec(1, 1)
ax = gs.subplots()
ax.plot(range(len(num_violations)), num_violations, label="joint")
ax.plot(
    range(len(num_violations_independent)),
    num_violations_independent,
    label="independent",
)
ax.set_xlim([0, 60])
ax.legend()
plt.savefig("./figures/num_constraint_violations.pdf")

fig = plt.figure()
gs = fig.add_gridspec(1, 1)
ax = gs.subplots()
ax.plot(range(len(extrinsic_return)), extrinsic_return, label="joint extrinsic return")
ax.plot(
    range(len(extrinsic_return_independent)),
    extrinsic_return_independent,
    label="independent extrinsic return",
)
ax.set_xlim([0, 60])
ax.legend()
plt.savefig("./figures/extrinsic_return.pdf")
# plt.show()

# api = wandb.Api()


# saved_runs = OmegaConf.load("./saved_runs.yaml")
# run = api.run(saved_runs.joint_gating.id)
# print(saved_runs)
# run = api.run(saved_runs.joint_gating.id)
# # metrics_dataframe = run.history()
# metrics_dataframe = run.history(pandas=False)
# print(len(metrics_dataframe))
# # metrics_dataframe.to_csv("metrics.csv")
# num_episodes_with_constraint_violations, num_constraint_violations = [], []
# extrinsic_return = []
# for step in metrics_dataframe:
#     try:
#         num_episodes_with_constraint_violations.append(
#             step["Num episodes with constraint violations"]
#         )
#         extrinsic_return.append(step["Extrinsic return"])
#         # num_constraint_violations.append(step["Num constraint violations"])
#     except KeyError:
#         pass


# run = api.run(saved_runs.independent_gating.id)
# metrics_dataframe = run.history(pandas=False)
# print(len(metrics_dataframe))
# num_episodes_with_constraint_violations_independent = []
# num_constraint_violations_independent = []
# extrinsic_return_independent = []
# # for step in run.history():
# for step in metrics_dataframe:
#     try:
#         num_episodes_with_constraint_violations_independent.append(
#             step["Num episodes with constraint violations"]
#         )
#         extrinsic_return_independent.append(step["Extrinsic return"])
#         # print(step)
#         # break
#         # num_constraint_violations_independent.append(step["Num constraint violations"])
#     except KeyError:
#         pass

# print(num_episodes_with_constraint_violations_independent)
# print(num_episodes_with_constraint_violations)

# plt.plot(
#     range(len(num_episodes_with_constraint_violations)),
#     num_episodes_with_constraint_violations,
#     label="joint",
# )
# plt.plot(
#     range(len(num_episodes_with_constraint_violations_independent)),
#     num_episodes_with_constraint_violations_independent,
#     label="independent",
# )
# plt.legend()

# plt.plot(
#     range(len(num_episodes_with_constraint_violations)),
#     num_episodes_with_constraint_violations,
#     label="joint",
# )
# plt.plot(
#     range(len(num_episodes_with_constraint_violations_independent)),
#     num_episodes_with_constraint_violations_independent,
#     label="independent",
# )

# # plt.plot(
# #     range(len(num_constraint_violations)),
# #     num_constraint_violations,
# #     label="joint",
# # )
# # plt.plot(
# #     range(len(num_constraint_violations_independent)),
# #     num_constraint_violations_independent,
# #     label="independent",
# # )
# # plt.legend()

# # print(run.history()["_step"])
# plt.show()
# # if run.state == "finished":
# # for i, row in run.history().iterrows():
# #     print(row["_timestamp"], row["loss"])
