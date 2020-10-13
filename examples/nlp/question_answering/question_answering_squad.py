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


import os

<<<<<<< HEAD
import numpy as np

import nemo.collections.nlp as nemo_nlp
import nemo.collections.nlp.data.tokenizers.tokenizer_utils
import nemo.core as nemo_core
from nemo import logging
from nemo.collections.nlp.callbacks.qa_squad_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.utils.lr_policies import get_lr_policy


def parse_args():
    parser = argparse.ArgumentParser(description="Squad_with_pretrained_BERT")
    parser.add_argument(
        "--train_file", type=str, help="The training data file. Should be *.json",
    )
    parser.add_argument(
        "--eval_file", type=str, help="The evaluation data file. Should be *.json",
    )
    parser.add_argument(
        "--test_file", type=str, help="The test data file. Should be *.json. Does not need to contain ground truth",
    )
    parser.add_argument(
        '--pretrained_model_name',
        default='roberta-base',
        type=str,
        help='Name of the pre-trained model',
        choices=nemo_nlp.nm.trainables.get_pretrained_lm_models_list(),
    )
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Checkpoint directory for inference.")
    parser.add_argument(
        "--bert_checkpoint", default=None, type=str, help="Path to BERT encoder checkpoint for finetuning."
    )
    parser.add_argument(
        "--head_checkpoint", default=None, type=str, help="Path to BERT QA head checkpoint for finetuning."
    )
    parser.add_argument("--bert_config", default=None, type=str, help="Path to bert config file in json format")
    parser.add_argument(
        "--tokenizer_model",
        default=None,
        type=str,
        help="Path to pretrained tokenizer model, only used if --tokenizer is sentencepiece",
    )
    parser.add_argument(
        "--tokenizer",
        default="nemobert",
        type=str,
        choices=["nemobert", "sentencepiece"],
        help="tokenizer to use, only relevant when using custom pretrained checkpoint.",
    )
    parser.add_argument("--optimizer", default="adam_w", type=str, help="Optimizer kind")
    parser.add_argument("--vocab_file", default=None, type=str, help="Path to the vocab file.")
    parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
    parser.add_argument("--lr", default=3e-5, type=float, help="The initial learning rate.")
    parser.add_argument("--lr_warmup_proportion", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--num_epochs", default=2, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If specified overrides --num_epochs.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training/evaluation.")
    parser.add_argument("--grad_norm_clip", type=float, default=-1, help="gradient clipping")
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--mode",
        default="train_eval",
        choices=["train", "train_eval", "eval", "test"],
        help="Mode of model usage. When using test mode the script is running inference on the data, i.e. no ground-truth labels are required in the dataset.",
    )
    parser.add_argument(
        "--no_data_cache", action='store_true', help="When specified do not load and store cache preprocessed data.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. "
        "Questions longer than this will be truncated to "
        "this length.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after "
        "WordPiece tokenization. Sequences longer than this "
        "will be truncated, and sequences shorter than this "
        " will be padded.",
    )
    parser.add_argument("--num_gpus", default=1, type=int, help="Number of GPUs")
    parser.add_argument(
        "--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"], help="01/02 to enable mixed precision"
    )
    parser.add_argument("--local_rank", type=int, default=None, help="For distributed training: local_rank")
    parser.add_argument(
        "--work_dir",
        default='output_squad',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--save_epoch_freq",
        default=1,
        type=int,
        help="Frequency of saving checkpoint '-1' - epoch checkpoint won't be saved",
    )
    parser.add_argument(
        "--save_step_freq",
        default=-1,
        type=int,
        help="Frequency of saving checkpoint '-1' - epoch checkpoint won't be saved",
    )
    parser.add_argument("--train_step_freq", default=100, type=int, help="Frequency of printing training loss")
    parser.add_argument(
        "--eval_step_freq", default=500, type=int, help="Frequency of evaluation during training on evaluation data"
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the examples contain some that do not have an answer.",
    )
    parser.add_argument(
        '--null_score_diff_threshold',
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    parser.add_argument(
        "--n_best_size", default=20, type=int, help="The total number of n-best predictions to generate at testing.",
    )
    parser.add_argument("--batches_per_step", default=1, type=int, help="Number of iterations per step.")
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be "
        "generated. This is needed because the start "
        "and end predictions are not conditioned "
        "on one another.",
    )
    parser.add_argument(
        "--output_prediction_file",
        type=str,
        required=False,
        default="predictions.json",
        help="File to write predictions to. Only in evaluation or test mode.",
    )
    parser.add_argument(
        "--output_nbest_file",
        type=str,
        required=False,
        default="nbest.json",
        help="File to write nbest predictions to. Only in evaluation or test mode.",
    )
    args = parser.parse_args()
    return args


def create_pipeline(
    data_file,
    model,
    head,
    max_query_length,
    max_seq_length,
    doc_stride,
    batch_size,
    version_2_with_negative,
    mode,
    num_gpus=1,
    batches_per_step=1,
    loss_fn=None,
    use_data_cache=True,
):
    data_layer = nemo_nlp.nm.data_layers.BertQuestionAnsweringDataLayer(
        mode=mode,
        version_2_with_negative=version_2_with_negative,
        batch_size=batch_size,
        tokenizer=tokenizer,
        data_file=data_file,
        max_query_length=max_query_length,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        shuffle="train" in mode,
        use_cache=use_data_cache,
    )

    input_data = data_layer()

    hidden_states = model(
        input_ids=input_data.input_ids, token_type_ids=input_data.input_type_ids, attention_mask=input_data.input_mask
    )

    qa_output = head(hidden_states=hidden_states)

    steps_per_epoch = len(data_layer) // (batch_size * num_gpus * batches_per_step)

    if mode == "test":
        return (
            steps_per_epoch,
            [input_data.unique_ids, qa_output],
            data_layer,
        )
    else:
        loss_output = loss_fn(
            logits=qa_output, start_positions=input_data.start_positions, end_positions=input_data.end_positions
        )

        return (
            loss_output.loss,
            steps_per_epoch,
            [input_data.unique_ids, loss_output.start_logits, loss_output.end_logits],
            data_layer,
        )


if __name__ == "__main__":
    args = parse_args()

    if "train" in args.mode:
        if not os.path.exists(args.train_file):
            raise FileNotFoundError(
                "train data not found. Datasets can be obtained using examples/nlp/question_answering/get_squad.py"
            )
    if "eval" in args.mode:
        if not os.path.exists(args.eval_file):
            raise FileNotFoundError(
                "eval data not found. Datasets can be obtained using examples/nlp/question_answering/get_squad.py"
            )
    if "test" in args.mode:
        if not os.path.exists(args.test_file):
            raise FileNotFoundError(
                "test data not found. Datasets can be obtained using examples/nlp/question_answering/get_squad.py"
            )

    # Instantiate neural factory with supported backend
    nf = nemo_core.NeuralModuleFactory(
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        log_dir=args.work_dir,
        create_tb_writer=True,
        files_to_copy=[__file__],
        add_time_to_log_dir=False,
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

    qa_head = nemo_nlp.nm.trainables.TokenClassifier(
        hidden_size=hidden_size, num_classes=2, num_layers=1, log_softmax=False, name="TokenClassifier"
    )
    squad_loss = nemo_nlp.nm.losses.SpanningLoss()

    if args.head_checkpoint is not None:
        qa_head.restore_from(args.head_checkpoint)

    if "train" in args.mode:
        train_loss, train_steps_per_epoch, _, _ = create_pipeline(
            data_file=args.train_file,
            model=model,
            head=qa_head,
            loss_fn=squad_loss,
            max_query_length=args.max_query_length,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            batch_size=args.batch_size,
            version_2_with_negative=args.version_2_with_negative,
            num_gpus=args.num_gpus,
            batches_per_step=args.batches_per_step,
            mode="train",
            use_data_cache=not args.no_data_cache,
        )
    if "eval" in args.mode:
        _, _, eval_output, eval_data_layer = create_pipeline(
            data_file=args.eval_file,
            model=model,
            head=qa_head,
            loss_fn=squad_loss,
            max_query_length=args.max_query_length,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            batch_size=args.batch_size,
            version_2_with_negative=args.version_2_with_negative,
            num_gpus=args.num_gpus,
            batches_per_step=args.batches_per_step,
            mode="eval",
            use_data_cache=not args.no_data_cache,
        )
    if "test" in args.mode:
        _, eval_output, test_data_layer = create_pipeline(
            data_file=args.test_file,
            model=model,
            head=qa_head,
            max_query_length=args.max_query_length,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            batch_size=args.batch_size,
            version_2_with_negative=args.version_2_with_negative,
            num_gpus=args.num_gpus,
            batches_per_step=args.batches_per_step,
            mode="test",
            use_data_cache=not args.no_data_cache,
        )

    if "train" in args.mode:
        logging.info(f"steps_per_epoch = {train_steps_per_epoch}")
        train_callback = nemo_core.SimpleLossLoggerCallback(
            tensors=[train_loss],
            print_func=lambda x: logging.info("Loss: {:.3f}".format(x[0].item())),
            get_tb_values=lambda x: [["loss", x[0]]],
            step_freq=args.train_step_freq,
            tb_writer=nf.tb_writer,
        )
        ckpt_callback = nemo_core.CheckpointCallback(
            folder=nf.checkpoint_dir, epoch_freq=args.save_epoch_freq, step_freq=args.save_step_freq
        )
        callbacks = [train_callback, ckpt_callback]
        if "eval" in args.mode:
            eval_callback = nemo_core.EvaluatorCallback(
                eval_tensors=eval_output,
                user_iter_callback=lambda x, y: eval_iter_callback(x, y),
                user_epochs_done_callback=lambda x: eval_epochs_done_callback(
                    x,
                    eval_data_layer=eval_data_layer,
                    do_lower_case=args.do_lower_case,
                    n_best_size=args.n_best_size,
                    max_answer_length=args.max_answer_length,
                    version_2_with_negative=args.version_2_with_negative,
                    null_score_diff_threshold=args.null_score_diff_threshold,
                ),
                tb_writer=nf.tb_writer,
                eval_step=args.eval_step_freq,
            )
            callbacks.append(eval_callback)

        optimization_params = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
        if args.max_steps < 0:
            total_steps = args.num_epochs * train_steps_per_epoch
            optimization_params['num_epochs'] = args.num_epochs
        else:
            total_steps = args.max_steps
            optimization_params['max_steps'] = args.max_steps
=======
import pytorch_lightning as pl
from omegaconf import DictConfig
>>>>>>> fd98a89adf80012987851a2cd3c3f4dc63bb8db6

from nemo.collections.nlp.models.question_answering.qa_model import QAModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="question_answering_squad_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config: {cfg.pretty()}')
    trainer = pl.Trainer(**cfg.trainer)
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    infer_datasets = [cfg.model.validation_ds, cfg.model.test_ds]
    for infer_dataset in infer_datasets:
        if infer_dataset.output_prediction_file is not None:
            infer_dataset.output_prediction_file = os.path.join(log_dir, infer_dataset.output_prediction_file)
        if infer_dataset.output_nbest_file is not None:
            infer_dataset.output_nbest_file = os.path.join(log_dir, infer_dataset.output_nbest_file)

    question_answering_model = QAModel(cfg.model, trainer=trainer)
    trainer.fit(question_answering_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.file is not None:
        gpu = 1 if cfg.trainer.gpus != 0 else 0
        trainer = pl.Trainer(gpus=gpu)
        if question_answering_model.prepare_test(trainer):
            trainer.test(question_answering_model)

    if cfg.model.nemo_path:
        question_answering_model.save_to(cfg.model.nemo_path)


if __name__ == '__main__':
    main()
