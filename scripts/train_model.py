import os
from absl import app, flags, logging
from accelerate import Accelerator
from tensorboardX import global_writer
from tensorboardX.global_writer import GlobalSummaryWriter
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler
import src.generation_patch  # noqa: F401, E501
import src.metrics as metrics
import src.utils as utils
from scripts.commonsense_qa import CommonSenseQADataset  # noqa: F401, E501
from scripts.numbers_data import ArithmethicDataset  # noqa: F401, E501
from scripts.parity_data import ParityDataset  # noqa: F401, E501
from src.postfix_tuner import (  # noqa: F401, E501
    GPT2PostfixLM,
    GPTNeoPostfixLM,
    GPTPostfixMixin,
)
from src.prompt_coder import (  # noqa: F401, E501
    GPT2PromptCoderLM,
    GPTJPromptCoderLM,
    GPTNeoPromptCoderLM,
    GPTPromptCoderMixin,
)
from src.prompt_tuner import (  # noqa: F401, E501
    GPT2PromptTuningCoderLM,
    GPT2PromptTuningLM,
    GPT2PromptTuningPostfixLM,
    GPTJPromptTuningCoderLM,
    GPTJPromptTuningLM,
    GPTJPromptTuningPostfixLM,
    GPTNeoPromptTuningCoderLM,
    GPTNeoPromptTuningLM,
    GPTNeoPromptTuningPostfixLM,
)


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_epochs", 15, help="Number of training epochs")

flags.DEFINE_integer("batch_size", 8, help="Batch size")

flags.DEFINE_integer("max_train_steps", 10, help="Number of training epochs")

flags.DEFINE_integer(
    "num_warmup_steps", 0, help="Number of warmup steps for lr scheduler"
)

flags.DEFINE_integer("n_prompt_tokens", None, help="Number of prompt_tokens")

flags.DEFINE_integer("n_coder_steps", None, help="Number of coder steps")

flags.DEFINE_boolean("init_from_vocab", True, help="Init prompt tokens from vocab")

flags.DEFINE_float("learning_rate", 0.001, help="Learning rate")

flags.DEFINE_float("weight_decay", 0.001, help="weight decay parameter for optimizer")

flags.DEFINE_string("lr_scheduler_type", "linear", help="Learning rate scheduler type")

flags.DEFINE_string("model", "gpt-2", help="gpt-2")

flags.DEFINE_string("model_type", "PromptTuningCoderLM", help="Which gadget to use")

flags.DEFINE_string("dataset", "ArithmethicDataset", help="dataset to train on")

flags.DEFINE_string("logdir", ".logs/", help="logdir for tensorboard logs")

flags.DEFINE_string("expdir", "exps/", help="logdir for tensorboard logs")

flags.DEFINE_string("suffix", None, help="suffix for exps")

flags.DEFINE_string(
    "resume_from_checkpoint", None, help="resume exps from a checkpoint"
)

flags.DEFINE_integer("seed", 0, help="Random generator seed")

flags.DEFINE_integer("gaccum", 1, help="gradient accumulation")

flags.DEFINE_integer("padding_idx", None, help="Padding idx")

flags.DEFINE_integer("evaluate_every", 100, help="Padding idx")

flags.DEFINE_boolean("disable_tqdm", False, help="Disable tqdm")

flags.DEFINE_boolean("parallelize", False, help="Model parallelize")

flags.DEFINE_integer("N_per_digit", 200, help="how many trials per digit")


def _experiment_suffix(FLAGS):
    model_name = FLAGS.model.replace("/", "_")
    data_name = FLAGS.dataset
    suffix = "_".join((model_name, data_name, FLAGS.model_type))
    return suffix


def _get_info_str(FLAGS):
    infostr = [f"- {k}: {v.value}" for k, v in FLAGS.__flags.items()]
    infostr = "   \n".join(infostr)
    return infostr


def get_model(FLAGS):
    if FLAGS.model_type == "FineTuning":
        model_type = "PromptCoderLM"
    else:
        model_type = FLAGS.model_type

    if "gpt2" in FLAGS.model:
        ModelType = eval("GPT2" + model_type)
    elif "Neo" in FLAGS.model:
        ModelType = eval("GPTNeo" + model_type)
    elif "j" in FLAGS.model:
        ModelType = eval("GPTJ" + model_type)
    else:
        raise ValueError(f"Unsupported model {ModelType}")

    kwargs = {
        "initialize_from_vocab": FLAGS.init_from_vocab,
        "padding_idx": FLAGS.padding_idx,
    }

    params_to_optimize = []
    if "Prompt" in FLAGS.model_type:
        params_to_optimize.append("soft_prompt")
        kwargs["n_tokens"] = FLAGS.n_prompt_tokens

    if "Postfix" in FLAGS.model_type or "Coder" in FLAGS.model_type:
        params_to_optimize.append("coder")
        kwargs["n_steps"] = FLAGS.n_coder_steps

    if len(params_to_optimize) == 0:
        params_to_optimize.append("")

    model = ModelType.from_pretrained(FLAGS.model, **kwargs)

    logging.info(str(ModelType))

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for name, p in model.named_parameters()
                if any((s in name for s in params_to_optimize))
            ],
            "weight_decay": FLAGS.weight_decay,
            "names": [
                name
                for name, p in model.named_parameters()
                if any((s in name for s in params_to_optimize))
            ],
        }
    ]
    logging.info(f"Params to optimize: {optimizer_grouped_parameters[0]['names']}")

    optimizer = AdamW(optimizer_grouped_parameters, lr=FLAGS.learning_rate)

    lr_scheduler = get_scheduler(
        name=FLAGS.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=FLAGS.num_warmup_steps,
        num_training_steps=FLAGS.max_train_steps,
    )

    return model, optimizer, lr_scheduler


def get_data(FLAGS):
    DataType = eval(FLAGS.dataset)
    datasets = [
        DataType(split=s, seed=FLAGS.seed, N_per_digit=FLAGS.N_per_digit)
        for s in ("train", "dev", "test")
    ]
    return DataType, datasets


def get_tokenizer(FLAGS):
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model)
    if FLAGS.padding_idx is None:
        tokenizer.pad_token = tokenizer.eos_token
        FLAGS.padding_idx = tokenizer.pad_token_id
    else:
        raise ValueError("Extra pad token id is not implemented")

    return tokenizer


def initializer_logger(FLAGS):
    FLAGS.suffix = _experiment_suffix(FLAGS)
    global_writer._writer = GlobalSummaryWriter(
        logdir=FLAGS.logdir, filename_suffix=FLAGS.suffix
    )


def get_batch_loader(DataType, tokenizer, dataset):
    collate_fn = DataType.get_collate(tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=(dataset.split == "train"),
        collate_fn=collate_fn,
    )
    return loader


def get_checkpoint_folder(FLAGS):
    path = os.path.join(FLAGS.expdir, "checkpoints/")
    os.makedirs(path, exist_ok=True)
    return path


def train_eval_loop(model, tokenizer, dataloader, tag="val", iter=0):
    writer = GlobalSummaryWriter.getSummaryWriter()
    total_loss = 0.0
    total_count = 0.0
    for data in tqdm(dataloader, disable=FLAGS.disable_tqdm):
        input = dict(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            labels=data["labels"],
        )
        if isinstance(model, GPTPromptCoderMixin) or isinstance(model, GPTPostfixMixin):
            input["input_lengths"] = data["input_lengths"][0]
        output = model(**input)
        loss = output.loss
        token_count = (data["labels"] != -100).sum().item()
        total_loss += loss.item() * token_count
        total_count += token_count

    avg_loss = total_loss / total_count
    logging.info(f"eval/{tag}/loss/{iter}: {avg_loss}")
    writer.add_scalar(f"eval/{tag}/loss", avg_loss, iter)


def train_loop(
    model, tokenizer, optimizer, lr_scheduler, accelerator, dataloader, iter=0
):
    writer = GlobalSummaryWriter.getSummaryWriter()
    total_loss = 0.0
    total_count = 0.0
    optimizer.zero_grad()
    for step, data in enumerate(tqdm(dataloader, disable=FLAGS.disable_tqdm)):
        input = dict(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            labels=data["labels"],
        )

        if isinstance(model, GPTPromptCoderMixin) or isinstance(model, GPTPostfixMixin):
            input["input_lengths"] = data["input_lengths"][0]

        output = model(**input)
        loss = output.loss / FLAGS.gaccum
        accelerator.backward(loss)

        if (step + 1) % FLAGS.gaccum == 0:
            optimizer.step()
            optimizer.zero_grad()

        token_count = (data["labels"] != -100).sum().item()
        total_loss += loss.item() * token_count
        total_count += token_count

        if (step + 1) % FLAGS.evaluate_every == 0:
            generation_loop(
                model, tokenizer, dataloader, kgen=10, tag="train", iter=iter
            )

    lr_scheduler.step()
    avg_loss = total_loss / total_count
    logging.info(f"train/loss/{iter}: {avg_loss}")
    writer.add_scalar("train/loss", avg_loss, iter)


def generation_loop(
    model,
    tokenizer,
    dataloader,
    kgen=10,
    disable=False,
    iter=0,
    tag="val",
    log=True,
    **kwargs,
):

    writer = GlobalSummaryWriter.getSummaryWriter()
    if disable:
        prev_disable = model.disable
        model.disable = True

    if model.disable:
        logging.info("Coder is disabled in generation")

    string_outputs = []
    string_answers = []
    string_inputs = []

    for (t, data) in enumerate(dataloader):
        limit = data["input_lengths"][0]
        input = dict(
            input_ids=data["input_ids"][:, :limit],
            attention_mask=data["attention_mask"][:, :limit],
            max_length=128,
        )

        if isinstance(model, GPTPromptCoderMixin) or isinstance(model, GPTPostfixMixin):
            input["input_lengths"] = limit

        output = model.generate(**input)

        generations = tokenizer.batch_decode(output, skip_special_tokens=True)

        answers = tokenizer.batch_decode(
            data["input_ids"][:, limit:], skip_special_tokens=True
        )

        inputs = tokenizer.batch_decode(
            data["input_ids"][:, :limit], skip_special_tokens=True
        )

        for i in range(len(generations)):
            inp = inputs[i]
            ans = answers[i].split(".")[0].strip()
            gen = generations[i].replace(inp, "").split(".")[0].strip()
            if log:
                logging.info(f"Q:{inp}\tP: {gen}\tA: {ans}")
            string_outputs.append(gen)
            string_answers.append(ans)
            string_inputs.append(inp)

        if t == kgen:
            break

    accuracy = metrics.char_accuracy(string_outputs, string_answers)
    logging.info(f"eval/{tag}/accuracy/{iter}: {accuracy}")
    writer.add_scalar(f"eval/{tag}/accuracy", accuracy, iter)

    accuracy = metrics.exact_accuracy(string_outputs, string_answers)
    logging.info(f"eval/{tag}/exactmatch/{iter}: {accuracy}")
    writer.add_scalar(f"eval/{tag}/exactmatch", accuracy, iter)

    if disable:
        model.disable = prev_disable

    return (
        string_inputs,
        string_outputs,
        string_answers,
    )


def read_off_loop(
    model, tokenizer, dataloader, kgen=10, disable=False, iter=0, tag="val", **kwargs
):

    writer = GlobalSummaryWriter.getSummaryWriter()
    if disable:
        prev_disable = model.disable
        model.disable = True

    if model.disable:
        logging.info("Coder is disabled in generation")

    string_outputs = []
    string_answers = []
    string_inputs = []

    for (t, data) in enumerate(dataloader):
        limit = data["input_lengths"][0]
        input = dict(
            input_ids=data["input_ids"][:, :limit],
            attention_mask=data["attention_mask"][:, :limit],
            max_length=32,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

        if isinstance(model, GPTPromptCoderMixin) or isinstance(model, GPTPostfixMixin):
            input["input_lengths"] = limit

        output = model.generate(**input)
        prefix_generations = None
        postfix_generations = None

        if hasattr(model, "n_tokens") and model.n_tokens is not None:
            prefix_len = model.n_tokens
            prefix_states = output.hidden_states[0][-1][:, :prefix_len, :]
            prefix_states = prefix_states.to(device=model.lm_head.weight.device)
            prefix_logits = model.lm_head(prefix_states).contiguous()
            prefix_ids = prefix_logits.argmax(dim=-1)
            prefix_generations = tokenizer.batch_decode(
                prefix_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        if hasattr(model, "n_steps") and model.n_steps is not None:
            postfix_len = model.n_steps
            postfix_states = output.hidden_states[0][-1][:, -postfix_len:, :]
            postfix_states = postfix_states.to(device=model.lm_head.weight.device)
            postfix_logits = model.lm_head(postfix_states).contiguous()
            postfix_ids = postfix_logits.argmax(dim=-1)
            postfix_generations = tokenizer.batch_decode(
                postfix_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        generations = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

        answers = tokenizer.batch_decode(
            data["input_ids"][:, limit:], skip_special_tokens=True
        )

        inputs = tokenizer.batch_decode(
            data["input_ids"][:, :limit], skip_special_tokens=True
        )

        for i in range(len(generations)):
            inp = inputs[i]
            ans = answers[i].split(".")[0].strip()
            gen = generations[i].replace(inp, "").split(".")[0].strip()
            logging.info(f"Q:{inp}\tP: {gen}\tA: {ans}")
            if prefix_generations is not None:
                prefix = prefix_generations[i].strip()
                logging.info(f"prefix:{prefix}")
            if postfix_generations is not None:
                postfix = postfix_generations[i].strip()
                logging.info(f"postfix:{postfix}")

            string_outputs.append(gen)
            string_answers.append(ans)
            string_inputs.append(inp)

        if t == kgen:
            break

    accuracy = metrics.char_accuracy(string_outputs, string_answers)
    logging.info(f"eval/{tag}/accuracy/{iter}: {accuracy}")
    writer.add_scalar(f"eval/{tag}/accuracy", accuracy, iter)

    accuracy = metrics.exact_accuracy(string_outputs, string_answers)
    logging.info(f"eval/{tag}/exactmatch/{iter}: {accuracy}")
    writer.add_scalar(f"eval/{tag}/exactmatch", accuracy, iter)

    if disable:
        model.disable = prev_disable

    return (
        string_inputs,
        string_outputs,
        string_answers,
    )


def train(_):
    initializer_logger(FLAGS)
    writer = GlobalSummaryWriter.getSummaryWriter()
    infostr = _get_info_str(FLAGS)
    logging.info(infostr)
    writer.add_text("FLAGS", infostr, 0)
    # assert not (FLAGS.n_prompt_tokens and FLAGS.n_coder_steps)

    utils.set_seed(FLAGS.seed)

    DataType, datasets = get_data(FLAGS)
    logging.info(f"{[len(d) for d in datasets]}")
    tokenizer = get_tokenizer(FLAGS)
    model, optimizer, lr_scheduler = get_model(FLAGS)

    if FLAGS.resume_from_checkpoint is not None:
        model, optimizer, lr_scheduler = utils.resume(
            model, optimizer, lr_scheduler, FLAGS.resume_from_checkpoint
        )

    dataloaders = [get_batch_loader(DataType, tokenizer, d) for d in datasets]

    accelerator = Accelerator()
    device = accelerator.device
    logging.info(f"Using device {device}")

    model, optimizer, *dataloaders = accelerator.prepare(model, optimizer, *dataloaders)

    model.parallelize()

    for epoch in range(FLAGS.num_train_epochs):
        # Eval Loop
        model.eval()
        logging.info(f"Starting epoch {epoch}, running generations for validation: ")
        generation_loop(model, tokenizer, dataloaders[1], kgen=10, iter=epoch)
        logging.info(
            "Generation loop for validation set is completed\n"
            "Running generations for trianing set"
        )
        generation_loop(
            model, tokenizer, dataloaders[0], kgen=10, tag="train", iter=epoch
        )
        logging.info(
            "Generation loop for training set is completed\n"
            "Runing evaluation loop for validation set"
        )
        train_eval_loop(model, tokenizer, dataloaders[1], iter=epoch)
        # read_off_loop(model, tokenizer, dataloaders[1], kgen=10, iter=epoch)
        logging.info(f"Epoch {epoch} training starts")
        # Train Loop
        model.train()
        # model.gradient_checkpointing_enable()
        train_loop(
            model,
            tokenizer,
            optimizer,
            lr_scheduler,
            accelerator,
            dataloaders[0],
            iter=epoch,
        )
        # model.gradient_checkpointing_disable()

    logging.info("Running final evals for test set")
    checkpoint_dir = get_checkpoint_folder(FLAGS)
    logging.info(f"Checkpoint dir: {checkpoint_dir}")

    inputs, outputs, answers = generation_loop(
        model,
        tokenizer,
        dataloaders[-1],
        kgen=len(dataloaders[-1]),
        iter=FLAGS.num_train_epochs,
        log=False,
        tag="test",
    )

    utils.save_evals(
        inputs, outputs, answers, checkpoint_dir, iter=FLAGS.num_train_epochs
    )

    train_eval_loop(
        model, tokenizer, dataloaders[-1], iter=FLAGS.num_train_epochs, tag="test"
    )

    utils.save_model(
        model, optimizer, lr_scheduler, checkpoint_dir, iter=FLAGS.num_train_epochs
    )


if __name__ == "__main__":
    app.run(train)
