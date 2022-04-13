import glob
import re
import traceback
import numpy as np
from absl import app, flags


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_folder", None, help="Path of generations tsv file")

reg = re.compile("Q:(.*)\nAnswer Choices:\n((.|\n)*?)\t(.*)\t(.*)")


def main(_):
    fnames = glob.glob(f"{FLAGS.exp_folder}/**/*test.tsv", recursive=True)
    for fname in fnames:
        print(fname)
        try:
            with open(fname) as handle:
                results = reg.findall(handle.read())
                labels = []
                for res in results:
                    if res[-1] == res[-2]:
                        labels.append(True)
                    else:
                        answer_keys = res[-4].split("\n")
                        answers = [" ".join(key.split(" ")[1:]) for key in answer_keys]
                        # print(res[-1])
                        gold_answer = " ".join(res[-1].split(" ")[:-1])
                        # print(answers)
                        # print(gold_answer)
                        correct_index = answers.index(gold_answer)
                        del answers[correct_index]
                        starts_with_any = any([a.startswith(res[-2]) for a in answers])
                        if not starts_with_any and res[-1].startswith(res[-2]):
                            labels.append(True)
                        else:
                            labels.append(False)
                print(np.mean(labels))
        except Exception:
            print(traceback.format_exc())


if __name__ == "__main__":
    app.run(main)
