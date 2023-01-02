import glob
import json
import re
import traceback
from absl import app, flags
from src import metrics


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_folder", None, help="Path of generations tsv file")


def accuracy_stats_of_a_set(fname):
    length_histogram = {}
    try:
        accuracy = {}
        all_generations = []
        all_answers = []
        with open(fname) as f:
            lines = f.read().replace("\n\t", "\t").splitlines()
            for line in lines:
                input, gen, ans = line.strip().split("\t")
                input_x = input.split("plus")[0].strip()
                input_x = re.sub("[^0-9]", "", input_x)
                gen = re.sub("[^0-9]", "", gen)
                ans = re.sub("[^0-9]", "", ans)
                N_digits = len(input_x)
                if N_digits not in length_histogram:
                    length_histogram[N_digits] = ([], [], [])
                length_histogram[N_digits][0].append(gen)
                length_histogram[N_digits][1].append(ans)
                length_histogram[N_digits][2].append(input)
            for N_digits, (
                generations,
                answers,
                inputs,
            ) in length_histogram.items():
                all_generations += generations
                all_answers += answers
                char_accuracy = metrics.char_accuracy(generations, answers)
                exact_match = metrics.exact_accuracy(generations, answers)
                accuracy[str(N_digits)] = {
                    "char_accuracy": char_accuracy,
                    "exact_match": exact_match,
                    "length": len(generations),
                }

            char_accuracy = metrics.char_accuracy(all_generations, all_answers)
            exact_match = metrics.exact_accuracy(all_generations, all_answers)
            accuracy["all"] = {
                "char_accuracy": char_accuracy,
                "exact_match": exact_match,
                "length": len(all_generations),
            }
        return accuracy
    except Exception:
        print(traceback.format_exc())
        return None


def main(_):
    fnames = glob.glob(f"{FLAGS.exp_folder}/**/*.tsv", recursive=True)
    for fname in fnames:
        accuracy = accuracy_stats_of_a_set(fname)
        iid_sum = 0.0
        iid_total = 0.0
        print(accuracy.keys())
        for k in range(2, 8):
            k = str(k)
            if k in accuracy:
                iid_total += accuracy[k]["length"]
                iid_sum += accuracy[k]["exact_match"] * accuracy[k]["length"]
        iid_accuracy = iid_sum / iid_total
        print("iid_accuracy: ", iid_accuracy)
        ood_sum = 0.0
        ood_total = 0.0
        for k in (9, 10):
            k = str(k)
            if k in accuracy:
                print(f"{k}-D accuracy: ", accuracy[k]["exact_match"])

        print(fname)
        # print(json.dumps(accuracy, indent=2))


if __name__ == "__main__":
    app.run(main)
