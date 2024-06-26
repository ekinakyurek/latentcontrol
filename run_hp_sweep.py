import os
import subprocess
import time
from absl import app, flags, logging
import scripts.train_model  # noqa: F401


# python run_hp_sweep.py --gpus_to_use 0,1,2,3 --dataset ESNLIDataset --gaccum 4 --exp_folder exps/ensli_later
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_folder", default=None, help="main experiment directory"
)

flags.DEFINE_string("gpus_to_use", default=None, help="coma seperated gpu ids")


def assign_to_gpu(gpus, file):
    logging.info(f"waiting for empty gpu for {file}")
    while True:
        for (k, v) in gpus.items():
            if len(v) == 0:
                v.append(file)
                return k
            for i in range(len(v)):
                if os.path.isfile(v[i]) or os.path.isdir(v[i]):
                    del v[i]
                    v.append(file)
                    return k
        time.sleep(60)


def main(_):
    train_cmd = (
        "export PYTHONHASHSEED=0;python -u scripts/train_model.py "
        "--disable_tqdm "
        f"--max_generation_len={FLAGS.max_generation_len} "
        f"--gaccum={FLAGS.gaccum} "
        f"--N_per_digit={FLAGS.N_per_digit} "
        f"--batch_size={FLAGS.batch_size} "
    )

    if FLAGS.reversed_outputs:
        train_cmd += "--reversed_outputs "

    gpus = list(FLAGS.gpus_to_use.split("_"))
    gpus = {id: [] for id in gpus}
    print(f"gpus: {gpus}")
    exp_files = []

    for seed in range(1):
        for train_type in ("FineTuning",):

            for backbone in ("EleutherAI/gpt-j-6B",):

                for step_index, steps in enumerate((30,)):

                    # if train_type == "FineTuning" and step_index > 0:
                    #     continue

                    # if train_type == "PromptTuningPostfixLM" and steps > 30:
                    #     continue

                    for learning_rate in (0.0001,):

                        if train_type == "FineTuning":
                            local_exp_folder = os.path.join(
                                FLAGS.exp_folder,
                                f"seed_{seed}",
                                train_type,
                                backbone,
                                f"lr_{learning_rate}",
                            )
                        else:
                            local_exp_folder = os.path.join(
                                FLAGS.exp_folder,
                                f"seed_{seed}",
                                train_type,
                                backbone,
                                f"step_{steps}",
                                f"lr_{learning_rate}",
                            )

                        os.makedirs(local_exp_folder, exist_ok=True)

                        params = (
                            f"--model={backbone} "
                            f"--model_type={train_type} "
                            f"--seed={seed} "
                            f"--expdir={local_exp_folder} "
                            f"--logdir={local_exp_folder} "
                            f"--learning_rate={learning_rate} "
                            f"--dataset={FLAGS.dataset} "
                        )

                        if train_type != "FineTuning":
                            if "Prompt" in train_type:
                                params += f"--n_prompt_tokens={steps}  "
                            if "Postfix" in train_type or "Coder" in train_type:
                                params += f"--n_coder_steps={steps} "

                        output_file = os.path.join(
                            local_exp_folder, "checkpoints/", "iter-15-test.tsv"
                        )
                        gpu = assign_to_gpu(gpus, output_file)
                        gpu_str = f"export CUDA_VISIBLE_DEVICES={gpu};"
                        log_str = (
                            f" > {local_exp_folder}/log.txt 2>"
                            f" {local_exp_folder}/err.txt "
                        )
                        cmd = gpu_str + train_cmd + params + log_str
                        logging.info(f"RUN: {cmd}")
                        # time.sleep(120)
                        subprocess.Popen(cmd, shell=True)
                        exp_files.append(output_file)


if __name__ == "__main__":
    app.run(main)
