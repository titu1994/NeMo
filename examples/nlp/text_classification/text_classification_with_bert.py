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
<<<<<<< HEAD
# =============================================================================

import argparse
import math

import nemo
import nemo.collections.nlp as nemo_nlp
from nemo import logging
from nemo.collections.nlp.callbacks.text_classification_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.collections.nlp.data.datasets import TextClassificationDataDesc
from nemo.utils.lr_policies import get_lr_policy

# Parsing arguments
parser = argparse.ArgumentParser(description='Sentence classification with pretrained BERT')
parser.add_argument("--work_dir", default='outputs', type=str)
parser.add_argument(
    "--checkpoint_dir",
    default=None,
    type=str,
    help="The folder containing the checkpoints for the model to continue training",
)
parser.add_argument("--data_dir", required=True, type=str)
parser.add_argument(
    '--pretrained_model_name',
    default='roberta-base',
    type=str,
    help='Name of the pre-trained model',
    choices=nemo_nlp.nm.trainables.get_pretrained_lm_models_list(),
)
parser.add_argument("--bert_checkpoint", default=None, type=str, help="Path to pre-trained BERT checkpoint")
parser.add_argument("--bert_config", default=None, type=str, help="Path to bert config file in json format")
parser.add_argument(
    "--tokenizer",
    default="nemobert",
    type=str,
    choices=["nemobert", "sentencepiece"],
    help="tokenizer to use, only relevant when using custom pretrained checkpoint.",
)
parser.add_argument("--vocab_file", default=None, type=str, help="Path to the vocab file.")
parser.add_argument(
    "--tokenizer_model",
    default=None,
    type=str,
    help="Path to pretrained tokenizer model, only used if --tokenizer is sentencepiece",
)
parser.add_argument(
    "--do_lower_case",
    action='store_true',
    help="Whether to lower case the input text. True for uncased models, False for cased models. "
    + "For tokenizer only applicable when tokenizer is build with vocab file.",
)
parser.add_argument("--batch_size", default=32, type=int, help="Training and evaluation batch size")
parser.add_argument(
    "--max_seq_length",
    default=36,
    type=int,
    help="The maximum total input sequence length after tokenization.Sequences longer than this will be \
                    truncated, sequences shorter will be padded.",
)
parser.add_argument("--num_gpus", default=1, type=int, help="Number of GPUs")
parser.add_argument("--num_output_layers", default=2, type=int, help="Number of layers in the Classifier")
parser.add_argument("--num_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--num_train_samples", default=-1, type=int, help="Number of samples to use for training")
parser.add_argument("--num_eval_samples", default=-1, type=int, help="Number of samples to use for evaluation")
parser.add_argument("--optimizer_kind", default="adam", type=str, help="Optimizer kind")
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float, help="Learning rate warm up proportion")
parser.add_argument("--lr", default=2e-5, type=float, help="Initial learning rate")
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str, help="Learning rate policy")
parser.add_argument(
    "--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"], help="01/02 to enable mixed precision"
)
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay.")
parser.add_argument("--fc_dropout", default=0.1, type=float, help="Dropout rate")
parser.add_argument(
    "--use_cache", action='store_true', help="When specified loads and stores cache preprocessed data."
)
parser.add_argument("--train_file_prefix", default='train', type=str, help="train file prefix")
parser.add_argument("--eval_file_prefix", default='dev', type=str, help="eval file prefix")
parser.add_argument("--class_balancing", default="None", type=str, choices=["None", "weighted_loss"])
parser.add_argument(
    "--no_shuffle_data", action='store_false', dest="shuffle_data", help="Shuffle is enabled by default."
)
parser.add_argument("--save_epoch_freq", default=1, type=int, help="Epoch frequency of saving checkpoints")
parser.add_argument("--save_step_freq", default=-1, type=int, help="Step frequency of saving checkpoints")
parser.add_argument('--loss_step_freq', default=25, type=int, help='Frequency of printing loss')
parser.add_argument('--eval_step_freq', default=100, type=int, help='Frequency of evaluation')
parser.add_argument("--local_rank", default=None, type=int, help="For distributed training: local_rank")

args = parser.parse_args()

nf = nemo.core.NeuralModuleFactory(
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=args.work_dir,
    checkpoint_dir=args.checkpoint_dir,
    create_tb_writer=True,
    files_to_copy=[__file__],
    add_time_to_log_dir=True,
)

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

data_desc = TextClassificationDataDesc(data_dir=args.data_dir, modes=[args.train_file_prefix, args.eval_file_prefix])

# Create sentence classification loss on top
classifier = nemo_nlp.nm.trainables.SequenceClassifier(
    hidden_size=hidden_size,
    num_classes=data_desc.num_labels,
    dropout=args.fc_dropout,
    num_layers=args.num_output_layers,
    log_softmax=False,
)

if args.class_balancing == 'weighted_loss':
    # You may need to increase the number of epochs for convergence.
    loss_fn = nemo.backends.pytorch.common.CrossEntropyLossNM(weight=data_desc.class_weights)
else:
    loss_fn = nemo.backends.pytorch.common.CrossEntropyLossNM()


def create_pipeline(num_samples=-1, batch_size=32, num_gpus=1, mode='train', is_training=True):
    logging.info(f"Loading {mode} data...")
    data_file = f'{data_desc.data_dir}/{mode}.tsv'
    shuffle = args.shuffle_data if is_training else False
    data_layer = nemo_nlp.nm.data_layers.BertTextClassificationDataLayer(
        input_file=data_file,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        num_samples=num_samples,
        shuffle=shuffle,
        batch_size=batch_size,
        use_cache=args.use_cache,
    )

    ids, type_ids, input_mask, labels = data_layer()
    data_size = len(data_layer)

    if data_size < batch_size:
        logging.warning("Batch_size is larger than the dataset size")
        logging.warning("Reducing batch_size to dataset size")
        batch_size = data_size

    steps_per_epoch = math.ceil(data_size / (batch_size * num_gpus))

    hidden_states = model(input_ids=ids, token_type_ids=type_ids, attention_mask=input_mask)

    logits = classifier(hidden_states=hidden_states)
    loss = loss_fn(logits=logits, labels=labels)

    if is_training:
        tensors_to_evaluate = [loss, logits]
=======

"""
This script contains an example on how to train, evaluate and perform inference with the TextClassificationModel.
TextClassificationModel in NeMo supports text classification problems such as sentiment analysis or
domain/intent detection for dialogue systems, as long as the data follows the format specified below.

***Data format***
TextClassificationModel requires the data to be stored in TAB separated files (.tsv) with two columns of sentence and
label. Each line of the data file contains text sequences, where words are separated with spaces and label separated
with [TAB], i.e.:

[WORD][SPACE][WORD][SPACE][WORD][TAB][LABEL]

For example:

hide new secretions from the parental units[TAB]0
that loves its characters and communicates something rather beautiful about human nature[TAB]1
...

If your dataset is stored in another format, you need to convert it to this format to use the TextClassificationModel.


***Setting the configs***
The model and the PT trainer are defined in a config file which declares multiple important sections.
The most important ones are:
    model: All arguments that are related to the Model - language model, tokenizer, head classifier, optimizer,
            schedulers, and datasets/data loaders.
    trainer: Any argument to be passed to PyTorch Lightning including number of epochs, number of GPUs,
            precision level, etc.

This script uses the `/examples/nlp/text_classification/conf/text_classification_config.yaml` default config file
by default. You may update the config file from the file directly or by using the command line arguments.
Other option is to set another config file via command line arguments by `--config-name=CONFIG_FILE_PATH'.

You first need to set the num_classes in the config file which specifies the number of classes in the dataset.
Notice that some config lines, including `model.dataset.classes_num`, have `???` as their value, this means that values
for these fields are required to be specified by the user. We need to specify and set the `model.train_ds.file_name`,
`model.validation_ds.file_name`, and `model.test_ds.file_name` in the config file to the paths of the train, validation,
 and test files if they exist. We may do it by updating the config file or by setting them from the command line.


***How to run the script?***
For example the following would train a model for 50 epochs in 2 GPUs on a classification task with 2 classes:

# python text_classification_with_bert.py
        model.dataset.num_classes=2
        model.train_ds=PATH_TO_TRAIN_FILE
        model.validation_ds=PATH_TO_VAL_FILE
        trainer.max_epochs=50
        trainer.gpus=2

This script would also reload the last checkpoint after the training is done and does evaluation on the dev set,
then performs inference on some sample queries.

By default, this script uses examples/nlp/text_classification/conf/text_classifciation_config.py config file, and
you may update all the params in the config file from the command line. You may also use another config file like this:

# python text_classification_with_bert.py --config-name==PATH_TO_CONFIG_FILE
        model.dataset.num_classes=2
        model.train_ds=PATH_TO_TRAIN_FILE
        model.validation_ds=PATH_TO_VAL_FILE
        trainer.max_epochs=50
        trainer.gpus=2

"""
import os

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.text_classification import TextClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="text_classification_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'\nConfig Params:\n{cfg.pretty()}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if not cfg.model.train_ds.file_path:
        raise ValueError("'train_ds.file_path' need to be set for the training!")

    model = TextClassificationModel(cfg.model, trainer=trainer)
    logging.info("===========================================================================================")
    logging.info('Starting training...')
    trainer.fit(model)
    logging.info('Training finished!')
    logging.info("===========================================================================================")

    if cfg.model.nemo_path:
        model.save_to(cfg.model.nemo_path)
        logging.info(f'Model is saved into `.nemo` file: {cfg.model.nemo_path}')

    # We evaluate the trained model on the test set if test_ds is set in the config file
    if cfg.model.test_ds.file_path:
        logging.info("===========================================================================================")
        logging.info("Starting the testing of the trained model on test set...")
        # The latest checkpoint would be used, set ckpt_path to 'best' to use the best one
        trainer.test(model=model, ckpt_path=None, verbose=False)
        logging.info("Testing finished!")
        logging.info("===========================================================================================")

    # retrieve the path to the last checkpoint of the training
    if trainer.checkpoint_callback is not None:
        checkpoint_path = os.path.join(
            trainer.checkpoint_callback.dirpath, trainer.checkpoint_callback.prefix + "end.ckpt"
        )
    else:
        checkpoint_path = None
    """
    After model training is done, if you have saved the checkpoints, you can create the model from 
    the checkpoint again and evaluate it on a data file. 
    You need to set or pass the test dataloader, and also create a trainer for this.
    """
    if checkpoint_path and os.path.exists(checkpoint_path) and cfg.model.validation_ds.file_path:
        logging.info("===========================================================================================")
        logging.info("Starting the evaluating the the last checkpoint on a data file (validation set by default)...")
        # we use the the path of the checkpoint from last epoch from the training, you may update it to any checkpoint
        # Create an evaluation model and load the checkpoint
        eval_model = TextClassificationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

        # create a dataloader config for evaluation, the same data file provided in validation_ds is used here
        # file_path can get updated with any file
        eval_config = OmegaConf.create(
            {'file_path': cfg.model.validation_ds.file_path, 'batch_size': 64, 'shuffle': False}
        )
        eval_model.setup_test_data(test_data_config=eval_config)

        # a new trainer is created to show how to evaluate a checkpoint from an already trained model
        # create a copy of the trainer config and update it to be used for final evaluation
        eval_trainer_cfg = cfg.trainer.copy()

        # it is safer to perform evaluation on single GPU without ddp as we are creating second trainer in
        # the same script, and it can be a problem with multi-GPU training.
        # We also need to reset the environment variable PL_TRAINER_GPUS to prevent PT from initializing ddp.
        # When evaluation and training scripts are in separate files, no need for this resetting.
        eval_trainer_cfg.gpus = 1 if torch.cuda.is_available() else 0
        eval_trainer_cfg.distributed_backend = None
        eval_trainer = pl.Trainer(**eval_trainer_cfg)

        eval_trainer.test(model=eval_model, verbose=False)  # test_dataloaders=eval_dataloader,

        logging.info("Evaluation the last checkpoint finished!")
        logging.info("===========================================================================================")
>>>>>>> fd98a89adf80012987851a2cd3c3f4dc63bb8db6
    else:
        logging.info(
            "No file_path was set for validation_ds or no checkpoint was found, so final evaluation is skipped!"
        )

    if checkpoint_path and os.path.exists(checkpoint_path):
        # You may create a model from a saved chechpoint and use the model.infer() method to
        # perform inference on a list of queries. There is no need of any trainer for inference.
        logging.info("===========================================================================================")
        logging.info("Starting the inference on some sample queries...")
        queries = [
            'by the end of no such thing the audience , like beatrice , has a watchful affection for the monster .',
            'director rob marshall went out gunning to make a great one .',
            'uneasy mishmash of styles and genres .',
        ]

        # use the path of the last checkpoint from the training, you may update it to any other checkpoints
        infer_model = TextClassificationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

        # move the model to the desired device for inference
        # we move the model to "cuda" if available otherwise "cpu" would be used
        if torch.cuda.is_available():
            infer_model.to("cuda")
        else:
            infer_model.to("cpu")

        # max_seq_length=512 is the maximum length BERT supports.
        results = infer_model.classifytext(queries=queries, batch_size=16, max_seq_length=512)

        logging.info('The prediction results of some sample queries with the trained model:')
        for query, result in zip(queries, results):
            logging.info(f'Query : {query}')
            logging.info(f'Predicted label: {result}')

        logging.info("Inference finished!")
        logging.info("===========================================================================================")
    else:
        logging.info("Inference is skipped as no checkpoint was found from the training!")


if __name__ == '__main__':
    main()
