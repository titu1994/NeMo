# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

<<<<<<< HEAD
import nemo
import nemo.collections.nlp as nemo_nlp
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM, LossAggregatorNM
from nemo.collections.nlp.callbacks.punctuation_capitalization_callback import (
    eval_epochs_done_callback,
    eval_iter_callback,
)
from nemo.collections.nlp.data.datasets.datasets_utils import calc_class_weights
from nemo.collections.nlp.nm.data_layers import PunctuationCapitalizationDataLayer
from nemo.collections.nlp.nm.trainables import PunctCapitTokenClassifier
from nemo.utils.lr_policies import get_lr_policy

# Parsing arguments
parser = argparse.ArgumentParser(
    description="Punctuation and \
    capitalization model with pretrained BERT"
)
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=5, type=int)
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--lr", default=5e-5, type=float)
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
parser.add_argument("--grad_norm_clip", type=float, default=1, help="Gradient clipping")
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--optimizer_kind", default="adam", type=str)
parser.add_argument("--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--data_dir", default="/data", type=str)
parser.add_argument("--punct_num_fc_layers", default=3, type=int)
parser.add_argument("--capit_num_fc_layers", default=2, type=int)
parser.add_argument("--fc_dropout", default=0.1, type=float)
parser.add_argument("--ignore_start_end", action='store_false')
parser.add_argument("--ignore_extra_tokens", action='store_false')
parser.add_argument("--none_label", default='O', type=str)
parser.add_argument("--no_shuffle_data", action='store_false', dest="shuffle_data")
parser.add_argument(
    "--pretrained_model_name",
    default="bert-base-uncased",
    type=str,
    help="Name of the pre-trained model",
    choices=nemo_nlp.nm.trainables.get_pretrained_lm_models_list(),
)
parser.add_argument("--bert_checkpoint", default=None, type=str)
parser.add_argument("--bert_config", default=None, type=str, help="Path to bert config file in json format")
parser.add_argument("--punct_classifier_checkpoint", default=None, type=str)
parser.add_argument("--capit_classifier_checkpoint", default=None, type=str)
parser.add_argument(
    "--tokenizer_model",
    default=None,
    type=str,
    help="Path to pretrained tokenizer model, \
                    only used if --tokenizer is sentencepiece",
)
parser.add_argument(
    "--tokenizer",
    default="nemobert",
    type=str,
    choices=["nemobert", "sentencepiece"],
    help="tokenizer to use, \
                    only relevant when using custom pretrained checkpoint.",
)
parser.add_argument(
    "--vocab_file", default=None, type=str, help="Path to the vocab file. Required for pretrained Megatron models"
)
parser.add_argument(
    "--do_lower_case",
    action='store_true',
    help="Whether to lower case the input text. True for uncased models, False for cased models. "
    + "Only applicable when tokenizer is build with vocab file",
)
parser.add_argument(
    "--work_dir",
    default='output',
    type=str,
    help="The output directory where the model prediction\
                    and checkpoints will be written.",
)
parser.add_argument(
    "--checkpoints_to_keep", default=1, type=int, help="The number of last checkpoints to keep",
)
parser.add_argument(
    '--add_confusion_matrix',
    action='store_true',
    help='Calculates and plots confusion matrix. Increases evaluation time.',
)
parser.add_argument(
    "--checkpoint_dir",
    default=None,
    type=str,
    help="The folder containing the checkpoints for the model to continue training",
)
parser.add_argument(
    "--overwrite_processed_files", action='store_true', help="Whether to overwrite preprocessed data files"
)
parser.add_argument(
    "--save_epoch_freq",
    default=1,
    type=int,
    help="Frequency of saving checkpoint\
                    '-1' - step checkpoint won't be saved",
)
parser.add_argument(
    "--save_step_freq",
    default=200,
    type=int,
    help="Frequency of saving checkpoint \
                    '-1' - step checkpoint won't be saved",
)
parser.add_argument(
    "--eval_epoch_freq", default=1, type=int, help="Frequency of evaluation",
)
parser.add_argument(
    "--loss_log_freq", default=50, type=int, help="Frequency of logging loss values, '-1' - at the end of the epoch",
)
parser.add_argument(
    "--use_weighted_loss_punct",
    action='store_true',
    help="Flag to indicate whether to use weighted loss \
                    to mitigate classs unbalancing for the punctuation task",
)
parser.add_argument(
    "--wandb_project", default=None, type=str, help='Project name for tracking with Weights and Biases'
)
parser.add_argument(
    "--wandb_exp_name", default=None, type=str, help='Experiment name for tracking with Weights and Biases'
)
parser.add_argument("--punct_loss_weight", default=0.5, type=float, help="Punctuation task weight loss")
parser.add_argument(
    "--num_workers",
    default=2,
    type=int,
    help="Number of workers for data loading, -1 means set it automatically to the number of CPU cores",
)

parser.add_argument(
    "--enable_pin_memory", action="store_true", help="Enables the pin_memory feature of Pytroch's DataLoader",
)
args = parser.parse_args()
=======
from nemo.collections.nlp.models import PunctuationCapitalizationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


"""
To run this script and train the model from scratch, use:
    python punctuation_and_capitalization.py \
    model.dataset.data_dir=PATH_TO_DATA_DIR
>>>>>>> fd98a89adf80012987851a2cd3c3f4dc63bb8db6

To use one of the pretrained versions of the model, run:
    python punctuation_and_capitalization.py \
    pretrained_model=Punctuation_Capitalization_with_BERT

<<<<<<< HEAD
nf = nemo.core.NeuralModuleFactory(
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=args.work_dir,
    create_tb_writer=True,
    checkpoint_dir=args.checkpoint_dir,
    files_to_copy=[__file__],
    add_time_to_log_dir=True,
)

logging.info(f"{args}")
=======
To use one of the pretrained versions of the model and finetune it, run:
    python punctuation_and_capitalization.py \
    pretrained_model=Punctuation_Capitalization_with_BERT \
    model.dataset.data_dir=PATH_TO_DATA_DIR

More details on the task and data format could be found in tutorials/nlp/Punctuation_and_Capitalization.ipynb
"""
>>>>>>> fd98a89adf80012987851a2cd3c3f4dc63bb8db6


<<<<<<< HEAD
model = nemo_nlp.nm.trainables.get_pretrained_lm_model(
    pretrained_model_name=args.pretrained_model_name,
    config=args.bert_config,
    vocab=args.vocab_file,
    checkpoint=args.bert_checkpoint,
)

tokenizer = nemo.collections.nlp.data.tokenizers.get_tokenizer(
    tokenizer_name=args.tokenizer,
    pretrained_model_name=args.pretrained_model_name,
    tokenizer_model=args.tokenizer_model,
    vocab_file=args.vocab_file,
    do_lower_case=args.do_lower_case,
)

hidden_size = model.hidden_size


def create_pipeline(
    pad_label=args.none_label,
    max_seq_length=args.max_seq_length,
    batch_size=args.batch_size,
    num_gpus=args.num_gpus,
    mode='train',
    punct_label_ids=None,
    capit_label_ids=None,
    ignore_extra_tokens=args.ignore_extra_tokens,
    ignore_start_end=args.ignore_start_end,
    overwrite_processed_files=args.overwrite_processed_files,
    dropout=args.fc_dropout,
    punct_num_layers=args.punct_num_fc_layers,
    capit_num_layers=args.capit_num_fc_layers,
    classifier=PunctCapitTokenClassifier,
):

    logging.info(f"Loading {mode} data...")
    shuffle = args.shuffle_data if mode == 'train' else False

    text_file = f'{args.data_dir}/text_{mode}.txt'
    label_file = f'{args.data_dir}/labels_{mode}.txt'

    if not (os.path.exists(text_file) or (os.path.exists(label_file))):
        raise FileNotFoundError(
            f'{text_file} or {label_file} not found. \
           The data should be splitted into 2 files: text.txt and labels.txt. \
           Each line of the text.txt file contains text sequences, where words\
           are separated with spaces. The labels.txt file contains \
           corresponding labels for each word in text.txt, the labels are \
           separated with spaces. Each line of the files should follow the \
           format:  \
           [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and \
           [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).'
        )

    data_layer = PunctuationCapitalizationDataLayer(
        tokenizer=tokenizer,
        text_file=text_file,
        label_file=label_file,
        pad_label=pad_label,
        punct_label_ids=punct_label_ids,
        capit_label_ids=capit_label_ids,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        shuffle=shuffle,
        ignore_extra_tokens=ignore_extra_tokens,
        ignore_start_end=ignore_start_end,
        overwrite_processed_files=overwrite_processed_files,
        num_workers=args.num_workers,
        pin_memory=args.enable_pin_memory,
    )

    (input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, punct_labels, capit_labels) = data_layer()

    if mode == 'train':
        punct_label_ids = data_layer.dataset.punct_label_ids
        capit_label_ids = data_layer.dataset.capit_label_ids
        class_weights = None

        if args.use_weighted_loss_punct:
            logging.info(f"Using weighted loss for punctuation task")
            punct_label_freqs = data_layer.dataset.punct_label_frequencies
            class_weights = calc_class_weights(punct_label_freqs)

        classifier = classifier(
            hidden_size=hidden_size,
            punct_num_classes=len(punct_label_ids),
            capit_num_classes=len(capit_label_ids),
            dropout=dropout,
            punct_num_layers=punct_num_layers,
            capit_num_layers=capit_num_layers,
        )

        punct_loss = CrossEntropyLossNM(logits_ndim=3, weight=class_weights)
        capit_loss = CrossEntropyLossNM(logits_ndim=3)
        task_loss = LossAggregatorNM(num_inputs=2, weights=[args.punct_loss_weight, 1.0 - args.punct_loss_weight])

    hidden_states = model(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

    punct_logits, capit_logits = classifier(hidden_states=hidden_states)

    if mode == 'train':
        punct_loss = punct_loss(logits=punct_logits, labels=punct_labels, loss_mask=loss_mask)
        capit_loss = capit_loss(logits=capit_logits, labels=capit_labels, loss_mask=loss_mask)
        task_loss = task_loss(loss_1=punct_loss, loss_2=capit_loss)

        steps_per_epoch = len(data_layer) // (batch_size * num_gpus)

        losses = [task_loss, punct_loss, capit_loss]
        logits = [punct_logits, capit_logits]
        return losses, logits, steps_per_epoch, punct_label_ids, capit_label_ids, classifier
    else:
        tensors_to_evaluate = [punct_logits, capit_logits, punct_labels, capit_labels, subtokens_mask]
        return tensors_to_evaluate, data_layer


(losses, train_logits, steps_per_epoch, punct_label_ids, capit_label_ids, classifier) = create_pipeline()

eval_tensors, data_layer = create_pipeline(
    mode='dev', punct_label_ids=punct_label_ids, capit_label_ids=capit_label_ids, classifier=classifier
)

logging.info(f"steps_per_epoch = {steps_per_epoch}")


# Create trainer and execute training action
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=losses + train_logits,
    print_func=lambda x: logging.info("Loss: {:.3f}".format(x[0].item())),
    get_tb_values=lambda x: [["loss", x[0]]],
    step_freq=args.loss_log_freq if args.loss_log_freq > 0 else steps_per_epoch,
    tb_writer=nf.tb_writer,
)

graph_dir = f'{nf.work_dir}/graphs' if args.add_confusion_matrix else None

ckpt_callback = nemo.core.CheckpointCallback(
    folder=nf.checkpoint_dir,
    epoch_freq=args.save_epoch_freq,
    step_freq=args.save_step_freq,
    checkpoints_to_keep=args.checkpoints_to_keep,
)

callbacks = [train_callback, ckpt_callback]

if args.wandb_project is not None:
    wand_callback = nemo.core.WandbCallback(
        train_tensors=[losses[0]],
        wandb_name=args.wandb_exp_name,
        wandb_project=args.wandb_project,
        update_freq=args.loss_log_freq if args.loss_log_freq > 0 else steps_per_epoch,
        args=args,
    )
    callbacks.append(wand_callback)

eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=eval_tensors,
    user_iter_callback=lambda x, y: eval_iter_callback(x, y),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(x, punct_label_ids, capit_label_ids, graph_dir),
    tb_writer=nf.tb_writer,
    eval_step=args.eval_epoch_freq * steps_per_epoch,
    wandb_name=args.wandb_exp_name,
    wandb_project=args.wandb_project,
)
callbacks.append(eval_callback)
=======
@hydra_runner(config_path="conf", config_name="punctuation_capitalization_config")
def main(cfg: DictConfig) -> None:
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    do_training = True
    if not cfg.pretrained_model:
        logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
        model = PunctuationCapitalizationModel(cfg.model, trainer=trainer)
    else:
        logging.info(f'Loading pretrained model {cfg.pretrained_model}')
        model = PunctuationCapitalizationModel.from_pretrained(cfg.pretrained_model)
        data_dir = cfg.model.dataset.get('data_dir', None)
        if data_dir:
            # we can also do finetunining of the pretrained model but it will require
            # setting up train and validation Pytorch DataLoaders
            model.setup_training_data(data_dir=data_dir)
            # evaluation could be done on multiple files, use model.validation_ds.ds_items to specify multiple
            # data directories if needed
            model.setup_validation_data(data_dirs=data_dir)
            logging.info(f'Using config file of the pretrained model')
        else:
            do_training = False
            logging.info(
                f'Data dir should be specified for training/finetuning. '
                f'Using pretrained {cfg.pretrained_model} model weights and skipping finetuning.'
            )

    if do_training:
        trainer.fit(model)
        if cfg.model.nemo_path:
            model.save_to(cfg.model.nemo_path)

    logging.info(
        'During evaluation/testing, it is currently advisable to construct a new Trainer with single GPU '
        'and no DDP to obtain accurate results'
    )
    gpu = 1 if cfg.trainer.gpus != 0 else 0
    trainer = pl.Trainer(gpus=gpu)
    model.set_trainer(trainer)

    # run an inference on a few examples
    queries = [
        'we bought four shirts one pen and a mug from the nvidia gear store in santa clara',
        'what can i do for you today',
        'how are you',
    ]
    inference_results = model.add_punctuation_capitalization(queries)

    for query, result in zip(queries, inference_results):
        logging.info(f'Query : {query}')
        logging.info(f'Result: {result.strip()}\n')
>>>>>>> fd98a89adf80012987851a2cd3c3f4dc63bb8db6


<<<<<<< HEAD
nf.train(
    tensors_to_optimize=[losses[0]],
    callbacks=callbacks,
    lr_policy=lr_policy_fn,
    optimizer=args.optimizer_kind,
    optimization_params={
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_norm_clip": args.grad_norm_clip,
    },
)
=======
if __name__ == '__main__':
    main()
>>>>>>> fd98a89adf80012987851a2cd3c3f4dc63bb8db6
