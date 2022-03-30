from multiprocessing.sharedctypes import Value
import os
from absl import app
from absl import flags
from absl import logging
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers import get_scheduler

from scripts.numbers_data import ArithmethicDataset  # noqa: F401, E501
from scripts.parity_data import ParityDataset  # noqa: F401, E501
from src.prompt_coder import GPT2PromptCoderLM, GPTNeoPromptCoderLM, GPTPromptCoderMixin  # noqa: F401, E501
from src.postfix_tuner import GPT2PostfixLM, GPTNeoPostfixLM, GPTPostfixMixin  # noqa: F401, E501
from src.prompt_tuner import GPT2PromptTuningLM, GPTNeoPromptTuningLM  # noqa: F401, E501
from src.prompt_tuner import GPT2PromptTuningCoderLM, GPTNeoPromptTuningCoderLM  # noqa: F401, E501
from src.prompt_tuner import GPT2PromptTuningPostfixLM, GPTNeoPromptTuningPostfixLM  # noqa: F401, E501
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import global_writer
from tensorboardX.global_writer import GlobalSummaryWriter
import src.utils as utils
import src.metrics as metrics
import pdb

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_train_epochs', 15,
                     help='Number of training epochs')

flags.DEFINE_integer('batch_size', 8,
                     help='Batch size')

flags.DEFINE_integer('max_train_steps', 10,
                     help='Number of training epochs')

flags.DEFINE_integer('num_warmup_steps', 0,
                     help='Number of warmup steps for lr scheduler')

flags.DEFINE_integer('n_prompt_tokens', None,
                     help='Number of prompt_tokens')

flags.DEFINE_integer('n_coder_steps', None,
                     help='Number of coder steps')

flags.DEFINE_boolean('init_from_vocab', True,
                     help='Init prompt tokens from vocab')

flags.DEFINE_float('learning_rate', 0.001,
                   help='Learning rate')

flags.DEFINE_float('weight_decay', 0.001,
                   help='weight decay parameter for optimizer')

flags.DEFINE_string('lr_scheduler_type', 'linear',
                    help='Learning rate scheduler type')

flags.DEFINE_string('model', 'gpt-2',
                    help='gpt-2')

flags.DEFINE_string('model_type', 'PromptTuningCoderLM',
                    help='Which gadget to use')

flags.DEFINE_string('dataset', 'ArithmethicDataset',
                    help='dataset to train on')

flags.DEFINE_string('logdir', '.logs/',
                    help='logdir for tensorboard logs')

flags.DEFINE_string('expdir', 'exps/',
                    help='logdir for tensorboard logs')

flags.DEFINE_string('suffix', None,
                    help='suffix for exps')

flags.DEFINE_integer('seed', 0,
                     help="Random generator seed")

flags.DEFINE_integer('gaccum', 1,
                     help="gradient accumulation")

flags.DEFINE_integer('padding_idx', None,
                     help="Padding idx")

flags.DEFINE_boolean('disable_tqdm', False,
                     help='Disable tqdm')


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
    if FLAGS.model_type == 'FineTuning':
        model_type = 'PromptCoderLM'
    else:
        model_type = FLAGS.model_type

    if "gpt2" in FLAGS.model:
        ModelType = eval("GPT2" + model_type)
    else:
        ModelType = eval("GPTNeo" + model_type)

    kwargs = {'initialize_from_vocab': FLAGS.init_from_vocab,
              'padding_idx': FLAGS.padding_idx}

    params_to_optimize = []
    if 'Prompt' in FLAGS.model_type:
        params_to_optimize.append('soft_prompt')
        kwargs['n_tokens'] = FLAGS.n_prompt_tokens

    if 'Postfix' in FLAGS.model_type or 'Coder' in FLAGS.model_type:
        params_to_optimize.append('coder')
        kwargs['n_steps'] = FLAGS.n_coder_steps

    if len(params_to_optimize) == 0:
        params_to_optimize.append('')

    model = ModelType.from_pretrained(FLAGS.model, **kwargs)

    logging.info(str(ModelType))

    optimizer_grouped_parameters = [
        {
         "params": [p for name, p in model.named_parameters()
                    if any((s in name for s in params_to_optimize))],
         "weight_decay": FLAGS.weight_decay,
         "names": [name for name, p in model.named_parameters()
                   if any((s in name for s in params_to_optimize))],
        }]
    logging.info(f"Params to optimize: "
                 f"{optimizer_grouped_parameters[0]['names']}")

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
    datasets = [DataType(split=s, seed=FLAGS.seed) for s in ("train", "dev", "test")]
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
        logdir=FLAGS.logdir,
        filename_suffix=FLAGS.suffix
    )


def get_batch_loader(DataType, tokenizer, dataset):
    collate_fn = DataType.get_collate(tokenizer)
    loader = DataLoader(dataset,
                        batch_size=FLAGS.batch_size,
                        shuffle=(dataset.split == 'train'),
                        collate_fn=collate_fn)
    return loader


def get_checkpoint_folder(FLAGS):
    path = os.path.join(FLAGS.expdir, 'checkpoints/')
    os.makedirs(path, exist_ok=True)
    return path


def train_eval_loop(model, tokenizer, dataloader, tag="val", iter=0):
    writer = GlobalSummaryWriter.getSummaryWriter()
    total_loss = 0.0
    total_count = 0.0
    for data in tqdm(dataloader, disable=FLAGS.disable_tqdm):
        input = dict(input_ids=data['input_ids'],
                     attention_mask=data['attention_mask'],
                     labels=data['labels'])
        if isinstance(model, GPTPromptCoderMixin) or isinstance(model, GPTPostfixMixin):
            input['input_lengths'] = data['input_lengths'][0]
        output = model(**input)
        loss = output.loss
        token_count = (data['labels'] != -100).sum().item()
        total_loss += loss.item() * token_count
        total_count += token_count

    avg_loss = total_loss/total_count
    logging.info(f"eval/{tag}/loss/{iter}: {avg_loss}")
    writer.add_scalar(f"eval/{tag}/loss", avg_loss, iter)


def train_loop(model, tokenizer, optimizer, accelerator, dataloader, iter=0):
    writer = GlobalSummaryWriter.getSummaryWriter()
    total_loss = 0.0
    total_count = 0.0
    for data in tqdm(dataloader, disable=FLAGS.disable_tqdm):
        optimizer.zero_grad()
        input = dict(input_ids=data['input_ids'],
                     attention_mask=data['attention_mask'],
                     labels=data['labels'])
        if isinstance(model, GPTPromptCoderMixin) or isinstance(model, GPTPostfixMixin):
            input['input_lengths'] = data['input_lengths'][0]
        output = model(**input)
        loss = output.loss
        accelerator.backward(loss)
        optimizer.step()
        token_count = (data['labels'] != -100).sum().item()
        total_loss += loss.item() * token_count
        total_count += token_count
    avg_loss = total_loss/total_count
    logging.info(f"train/loss/{iter}: {avg_loss}")
    writer.add_scalar("train/loss", avg_loss, iter)


def generation_loop(model,
                    tokenizer,
                    dataloader,
                    kgen=10,
                    disable=False,
                    iter=0,
                    tag="val",
                    **kwargs):

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
        limit = data['input_lengths'][0]
        input = dict(input_ids=data['input_ids'][:, :limit],
                     attention_mask=data['attention_mask'][:, :limit],
                     max_length=32)

        if isinstance(model, GPTPromptCoderMixin) or isinstance(model, GPTPostfixMixin):
            input['input_lengths'] = limit

        output = model.generate(**input)

        generations = tokenizer.batch_decode(output,
                                             skip_special_tokens=True)

        answers = tokenizer.batch_decode(data['input_ids'][:, limit:],
                                         skip_special_tokens=True)

        inputs = tokenizer.batch_decode(data['input_ids'][:, :limit],
                                        skip_special_tokens=True)

        for i in range(len(generations)):
            inp = inputs[i]
            ans = answers[i].split('.')[0].strip()
            gen = generations[i].replace(inp, "").split('.')[0].strip()
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

    return string_inputs, string_outputs, string_answers,


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
    dataloaders = [get_batch_loader(DataType, tokenizer, d)
                   for d in datasets]

    accelerator = Accelerator()
    device = accelerator.device
    logging.info(f"Using device {device}")

    model, optimizer, *dataloaders = accelerator.prepare(model,
                                                         optimizer,
                                                         *dataloaders)

    for epoch in range(FLAGS.num_train_epochs):
        # Eval Loop
        model.eval()
        logging.info(f"Starting epoch {epoch}, running evals before")

        generation_loop(model,
                        tokenizer,
                        dataloaders[1],
                        kgen=10,
                        iter=epoch)

        train_eval_loop(model,
                        tokenizer,
                        dataloaders[1],
                        iter=epoch)

        logging.info(f"Epoch {epoch} training starts")
        # Train Loop
        model.train()
        train_loop(model,
                   tokenizer,
                   optimizer,
                   accelerator,
                   dataloaders[0],
                   iter=epoch)
        lr_scheduler.step()

    logging.info("Running final evals for test set")
    checkpoint_dir = get_checkpoint_folder(FLAGS)
    logging.info(f"Checkpoint dir: {checkpoint_dir}")

    inputs, outputs, answers = generation_loop(model,
                                               tokenizer,
                                               dataloaders[-1],
                                               kgen=len(dataloaders[-1]),
                                               iter=FLAGS.num_train_epochs,
                                               tag="test")

    utils.save_evals(inputs,
                     outputs,
                     answers,
                     checkpoint_dir,
                     iter=FLAGS.num_train_epochs)

    train_eval_loop(model,
                    tokenizer,
                    dataloaders[-1],
                    iter=FLAGS.num_train_epochs,
                    tag="test")

    utils.save_model(model,
                     optimizer,
                     lr_scheduler,
                     checkpoint_dir,
                     iter=FLAGS.num_train_epochs)


if __name__ == "__main__":
    app.run(train)
