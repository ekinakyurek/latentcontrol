import glob
import traceback
from absl import app, flags
from src import metrics


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_folder", None, help="Path of generations tsv file")


def main(_):
    fnames = glob.glob(f"{FLAGS.exp_folder}/**/*test.tsv", recursive=True)
    for fname in fnames:
        print(fname)
        try:
            generations = []
            answers = []
            with open(fname) as f:
                for line in f:
                    input, gen, ans = line.strip().split("\t")
                    if len(ans.replace(" ", "")) < 6:
                        generations.append(gen)
                        answers.append(ans)
            print("len examples: ", len(generations))
            print("char accuracy:", metrics.char_accuracy(generations, answers))
            print("exact match:", metrics.exact_accuracy(generations, answers))
        except Exception:
            print(traceback.format_exc())


if __name__ == "__main__":
    app.run(main)
