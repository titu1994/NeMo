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

import pytorch_lightning as pl
from omegaconf.listconfig import ListConfig
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import ExtractSpeakerEmbeddingsModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
<<<<<<< HEAD


def parse_args():
    parser = argparse.ArgumentParser(
        parents=[nm_argparse.NemoArgParser()], description='SpeakerRecognition', conflict_handler='resolve',
    )
    parser.set_defaults(
        checkpoint_dir=None,
        optimizer="novograd",
        batch_size=32,
        eval_batch_size=64,
        lr=0.01,
        weight_decay=0.001,
        amp_opt_level="O0",
        create_tb_writer=True,
    )

    # Overwrite default args
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        required=True,
        help="number of epochs to train. You should specify either num_epochs or max_steps",
    )
    parser.add_argument(
        "--model_config", type=str, required=True, help="model configuration file: model.yaml",
    )

    # Create new args
    parser.add_argument("--exp_name", default="SpkrReco_GramMatrix", type=str)
    parser.add_argument("--beta1", default=0.95, type=float)
    parser.add_argument("--beta2", default=0.5, type=float)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--load_dir", default=None, type=str)
    parser.add_argument("--synced_bn", action='store_true', help="Use synchronized batch norm")
    parser.add_argument("--synced_bn_groupsize", default=0, type=int)
    parser.add_argument("--emb_size", default=256, type=int)
    parser.add_argument("--print_freq", default=256, type=int)

    args = parser.parse_args()
    if args.max_steps is not None:
        raise ValueError("QuartzNet uses num_epochs instead of max_steps")

    return args


def construct_name(name, lr, batch_size, num_epochs, wd, optimizer, emb_size):
    return "{0}-lr_{1}-bs_{2}-e_{3}-wd_{4}-opt_{5}-embsize_{6}".format(
        name, lr, batch_size, num_epochs, wd, optimizer, emb_size
    )


def create_all_dags(args, neural_factory):
    '''
    creates train and eval dags as well as their callbacks
    returns train loss tensor and callbacks'''

    # parse the config files
    yaml = YAML(typ="safe")
    with open(args.model_config) as f:
        spkr_params = yaml.load(f)

    sample_rate = spkr_params['sample_rate']

    # Calculate num_workers for dataloader
    total_cpus = os.cpu_count()
    cpu_per_traindl = max(int(total_cpus / neural_factory.world_size), 1)

    # create separate data layers for eval
    # we need separate eval dags for separate eval datasets
    # but all other modules in these dags will be shared

    eval_dl_params = copy.deepcopy(spkr_params["AudioToSpeechLabelDataLayer"])
    eval_dl_params.update(spkr_params["AudioToSpeechLabelDataLayer"]["eval"])
    del eval_dl_params["train"]
    del eval_dl_params["eval"]
    eval_dl_params['shuffle'] = False  # To grab  the file names without changing data_layer

    data_layer_test = nemo_asr.AudioToSpeechLabelDataLayer(
        manifest_filepath=args.eval_datasets[0],
        labels=None,
        batch_size=args.batch_size,
        num_workers=cpu_per_traindl,
        **eval_dl_params,
        # normalize_transcripts=False
    )
    # create shared modules

    data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
        sample_rate=sample_rate, **spkr_params["AudioToMelSpectrogramPreprocessor"],
    )

    # (QuartzNet uses the Jasper baseline encoder and decoder)
    encoder = nemo_asr.JasperEncoder(**spkr_params["JasperEncoder"],)

    decoder = nemo_asr.JasperDecoderForSpkrClass(
        feat_in=spkr_params['JasperEncoder']['jasper'][-1]['filters'],
        num_classes=254,
        emb_sizes=spkr_params['JasperDecoderForSpkrClass']['emb_sizes'].split(','),
        pool_mode=spkr_params["JasperDecoderForSpkrClass"]['pool_mode'],
    )

    # --- Assemble Validation DAG --- #
    audio_signal_test, audio_len_test, label_test, _ = data_layer_test()

    processed_signal_test, processed_len_test = data_preprocessor(
        input_signal=audio_signal_test, length=audio_len_test
    )

    encoded_test, _ = encoder(audio_signal=processed_signal_test, length=processed_len_test)

    _, embeddings = decoder(encoder_output=encoded_test)

    return embeddings, label_test


def main():
    args = parse_args()

    print(args)

    name = construct_name(
        args.exp_name, args.lr, args.batch_size, args.num_epochs, args.weight_decay, args.optimizer, args.emb_size
    )
    work_dir = name
    if args.work_dir:
        work_dir = os.path.join(args.work_dir, name)

    # instantiate Neural Factory with supported backend
    neural_factory = nemo.core.NeuralModuleFactory(
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        log_dir=work_dir,
        checkpoint_dir=args.checkpoint_dir + "/" + args.exp_name,
        create_tb_writer=False,
        files_to_copy=[args.model_config, __file__],
        random_seed=42,
        cudnn_benchmark=args.cudnn_benchmark,
    )
    args.num_gpus = neural_factory.world_size

    args.checkpoint_dir = neural_factory.checkpoint_dir

    if args.local_rank is not None:
        logging.info('Doing ALL GPU')

    # build dags
    embeddings, label_test = create_all_dags(args, neural_factory)

    eval_tensors = neural_factory.infer(tensors=[embeddings, label_test], checkpoint_dir=args.checkpoint_dir)
    # inf_loss , inf_emb, inf_logits, inf_label = eval_tensors
    inf_emb, inf_label = eval_tensors
    whole_embs = []
    whole_labels = []
    manifest = open(args.eval_datasets[0], 'r').readlines()

    for line in manifest:
        line = line.strip()
        dic = json.loads(line)
        filename = dic['audio_filepath'].split('/')[-1]
        whole_labels.append(filename)

    for idx in range(len(inf_label)):
        whole_embs.extend(inf_emb[idx].numpy())

    embedding_dir = args.work_dir + './embeddings/'
    if not os.path.exists(embedding_dir):
        os.mkdir(embedding_dir)

    filename = os.path.basename(args.eval_datasets[0]).split('.')[0]
    name = embedding_dir + filename

    np.save(name + '.npy', np.asarray(whole_embs))
    np.save(name + '_labels.npy', np.asarray(whole_labels))
    logging.info("Saved embedding files to {}".format(embedding_dir))
=======
from nemo.utils.exp_manager import exp_manager

"""
To extract embeddings
Place pretrained model in ${EXP_DIR}/${EXP_NAME} with spkr.nemo
    python spkr_get_emb.py --config-path='conf' --config-name='SpeakerNet_verification_3x2x512.yaml' \
        +model.test_ds.manifest_filepath="<test_manifest_file>" \
        +model.test_ds.sample_rate=16000 \
        +model.test_ds.labels=null \
        +model.test_ds.batch_size=1 \
        +model.test_ds.shuffle=False \
        +model.test_ds.time_length=8 \
        exp_manager.exp_name=${EXP_NAME} \
        exp_manager.exp_dir=${EXP_DIR} \
        trainer.gpus=1 

See https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_recognition/Speaker_Recognition_Verification.ipynb for notebook tutorial
"""

seed_everything(42)


@hydra_runner(config_path="conf", config_name="config")
def main(cfg):

    logging.info(f'Hydra config: {cfg.pretty()}')
    if (isinstance(cfg.trainer.gpus, ListConfig) and len(cfg.trainer.gpus) > 1) or (
        isinstance(cfg.trainer.gpus, (int, str)) and int(cfg.trainer.gpus) > 1
    ):
        logging.info("changing gpus to 1 to minimize DDP issues while extracting embeddings")
        cfg.trainer.gpus = 1
        cfg.trainer.distributed_backend = None
    trainer = pl.Trainer(**cfg.trainer)
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    model_path = os.path.join(log_dir, '..', 'spkr.nemo')
    speaker_model = ExtractSpeakerEmbeddingsModel.restore_from(model_path)
    speaker_model.setup_test_data(cfg.model.test_ds)
    trainer.test(speaker_model)
>>>>>>> fd98a89adf80012987851a2cd3c3f4dc63bb8db6


if __name__ == '__main__':
    main()
