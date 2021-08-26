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

import copy
import os
from typing import Dict, Optional

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer_bpe import WERBPE
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes.common import PretrainedModelInfo
from nemo.collections.common.losses import NLLLoss, SmoothedCrossEntropyLoss
from nemo.collections.nlp.modules.common.transformer import BeamSearchSequenceGenerator
from nemo.utils import logging, model_utils

from nemo.core.classes.common import typecheck
from nemo.core.neural_types import (
    AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType, MaskType, ChannelType)


__all__ = ['EncDecAutoregModelBPE']


def lens_to_mask(lens, max_length):
    batch_size = lens.shape[0]
    mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]
    return mask


class EncDecAutoregModelBPE(EncDecCTCModel, ASRBPEMixin):
    """Encoder decoder CTC-based models with Byte Pair Encoding."""

    def __init__(self, cfg: DictConfig, trainer=None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        # Setup the tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Set the new vocabulary
        with open_dict(cfg):
            cfg.ctc_decoder.vocabulary = ListConfig(list(vocabulary.keys()))

        # Override number of classes if placeholder provided
        num_classes = cfg.ctc_decoder["num_classes"]

        if num_classes < 1:
            logging.info(
                "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                    num_classes, len(vocabulary)
                )
            )
            cfg.ctc_decoder["num_classes"] = len(vocabulary)

        super().__init__(cfg=cfg, trainer=trainer)

        with open_dict(self._cfg):
            if "feat_in" not in self._cfg.ctc_decoder or (
                not self._cfg.ctc_decoder.feat_in and hasattr(self.encoder, '_feat_out')
            ):
                self._cfg.ctc_decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self._cfg.ctc_decoder or not self._cfg.ctc_decoder.feat_in:
                raise ValueError("param feat_in of the decoder's config is not set!")

        self.beam_search = BeamSearchSequenceGenerator(
            embedding=self.decoder.embedding,
            decoder=self.decoder.decoder,
            log_softmax=self.log_softmax,
            max_sequence_length=self.decoder.max_sequence_length,
            beam_size=self._cfg.beam_size,
            bos=self.tokenizer.bos_id,
            pad=self.tokenizer.pad_id,
            eos=self.tokenizer.eos_id,
            len_pen=self._cfg.len_pen,
            max_delta_length=self._cfg.max_generation_delta,
        )

        self.loss_fn = SmoothedCrossEntropyLoss(
            pad_id=self.tokenizer.pad_id, label_smoothing=self._cfg.label_smoothing
        )
        
        self.ctc_decoder = EncDecCTCModel.from_config_dict(self._cfg.ctc_decoder)

        self.ctc_loss = CTCLoss(
            num_classes=self.ctc_decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )

        self.ctc_coef = self._cfg.get("ctc_coef", 0.5)

        # Setup metric objects
        self._wer = WERBPE(
            tokenizer=self.tokenizer,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            ctc_decode=False,
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_bpe_dataset(
                config=config,
                tokenizer=self.tokenizer,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = audio_to_text_dataset.get_bpe_dataset(
                config=config, tokenizer=self.tokenizer, augmentor=augmentor
            )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        dl_config = {
            'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': min(config['batch_size'], len(config['paths2audio_files'])),
            'shuffle': False,
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def change_vocabulary(self, new_tokenizer_dir: str, new_tokenizer_type: str):
        """
        Changes vocabulary of the tokenizer used during CTC decoding process.
        Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_tokenizer_dir: Path to the new tokenizer directory.
            new_tokenizer_type: Either `bpe` or `wpe`. `bpe` is used for SentencePiece tokenizers,
                whereas `wpe` is used for `BertTokenizer`.

        Returns: None

        """
        if not os.path.isdir(new_tokenizer_dir):
            raise NotADirectoryError(
                f'New tokenizer dir must be non-empty path to a directory. But I got: {new_tokenizer_dir}'
            )

        if new_tokenizer_type.lower() not in ('bpe', 'wpe'):
            raise ValueError(f'New tokenizer type must be either `bpe` or `wpe`')

        tokenizer_cfg = OmegaConf.create({'dir': new_tokenizer_dir, 'type': new_tokenizer_type})

        # Setup the tokenizer
        self._setup_tokenizer(tokenizer_cfg)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Set the new vocabulary
        decoder_config = copy.deepcopy(self.decoder.to_config_dict())
        decoder_config.vocabulary = ListConfig(list(vocabulary.keys()))

        decoder_num_classes = decoder_config['num_classes']

        # Override number of classes if placeholder provided
        logging.info(
            "\nReplacing old number of classes ({}) with new number of classes - {}".format(
                decoder_num_classes, len(vocabulary)
            )
        )

        decoder_config['num_classes'] = len(vocabulary)

        del self.decoder
        self.decoder = EncDecCTCModelBPE.from_config_dict(decoder_config)
        del self.loss
        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )
        self._wer = WERBPE(
            tokenizer=self.tokenizer,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            ctc_decode=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

        # Update config
        OmegaConf.set_struct(self._cfg.decoder, False)
        self._cfg.decoder = decoder_config
        OmegaConf.set_struct(self._cfg.decoder, True)

        logging.info(f"Changed tokenizer to {self.decoder.vocabulary} vocabulary.")

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            'transcript': NeuralType(('B', 'T'), LabelsType(), optional=True),
            'transcript_length': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "autoreg_log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "ctc_log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
            "src_hiddens": NeuralType(('B', 'T', 'D'), ChannelType()),
            "src_mask": NeuralType(('B', 'T'), MaskType()),
        }

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None,
        processed_signal=None, processed_signal_length=None,
        transcript=None, transcript_length=None,
    ):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        ctc_log_probs = self.ctc_decoder(encoder_output=encoded)
        greedy_predictions = ctc_log_probs.argmax(dim=-1, keepdim=False)
        
        src_hiddens = encoded.permute(0, 2, 1)
        src_mask = lens_to_mask(encoded_len, src_hiddens.shape[1]).to(src_hiddens.dtype)
        tgt_mask = lens_to_mask(transcript_length, transcript.shape[1]).to(transcript.dtype)

        tgt_hiddens = self.decoder(
            input_ids=transcript, decoder_mask=tgt_mask, encoder_embeddings=src_hiddens, encoder_mask=src_mask
        )
        log_probs = self.log_softmax(hidden_states=tgt_hiddens)

        return log_probs, ctc_log_probs, encoded_len, greedy_predictions, src_hiddens, src_mask

    def training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len = batch
        
        input_ids, labels = transcript[:, :-1], transcript[:, 1:]

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, ctc_log_probs, encoded_len, predictions, src_hiddens, src_mask = self.forward(
                processed_signal=signal, processed_signal_length=signal_len,
                transcript=input_ids, transcript_length=transcript_len
            )
        else:
            log_probs, ctc_log_probs, encoded_len, predictions, src_hiddens, src_mask = self.forward(
                input_signal=signal, input_signal_length=signal_len,
                transcript=input_ids, transcript_length=transcript_len
            )

        autoreg_loss = self.loss_fn(log_probs=log_probs, labels=labels)

        ctc_loss = self.ctc_loss(
            log_probs=ctc_log_probs, targets=transcript,
            input_lengths=encoded_len, target_lengths=transcript_len
        )

        loss_value = self.ctc_coef * ctc_loss + (1 - self.ctc_coef) * autoreg_loss

        tensorboard_logs = {'train_loss': loss_value, 'learning_rate': self._optimizer.param_groups[0]['lr']}

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        if (batch_nb + 1) % log_every_n_steps == 0:
            self._wer.update(
                predictions=predictions,
                targets=transcript,
                target_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            wer, _, _ = self._wer.compute()
            self._wer.reset()
            tensorboard_logs.update({'training_batch_wer': wer})

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        signal, signal_len, transcript, transcript_len = batch
        
        input_ids, labels = transcript[:, :-1], transcript[:, 1:]

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, _, encoded_len, predictions, src_hiddens, src_mask = self.forward(
                processed_signal=signal, processed_signal_length=signal_len,
                transcript=input_ids, transcript_length=transcript_len
            )
        else:
            log_probs, _, encoded_len, predictions, src_hiddens, src_mask = self.forward(
                input_signal=signal, input_signal_length=signal_len,
                transcript=input_ids, transcript_length=transcript_len
            )
            
        beam_hypotheses = self.beam_search(
            encoder_hidden_states=src_hiddens, encoder_input_mask=src_mask, return_beam_scores=False
        )

        loss_value = self.loss_fn(log_probs=log_probs, labels=labels)

        self._wer.update(
            predictions=beam_hypotheses, targets=transcript,
            target_lengths=transcript_len, predictions_lengths=encoded_len
        )
        wer, wer_num, wer_denom = self._wer.compute()
        self._wer.reset()
        return {
            'val_loss': loss_value,
            'val_wer_num': wer_num,
            'val_wer_denom': wer_denom,
            'val_wer': wer,
        }

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_256",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_256",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_256/versions/1.0.0rc1/files/stt_en_citrinet_256.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_512",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_512",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_512/versions/1.0.0rc1/files/stt_en_citrinet_512.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_1024",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_1024",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_1024/versions/1.0.0rc1/files/stt_en_citrinet_1024.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_256_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_256_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_256_gamma_0_25/versions/1.0.0/files/stt_en_citrinet_256_gamma_0_25.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_512_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_512_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_512_gamma_0_25/versions/1.0.0/files/stt_en_citrinet_512_gamma_0_25.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_1024_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_1024_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_1024_gamma_0_25/versions/1.0.0/files/stt_en_citrinet_1024_gamma_0_25.nemo",
        )

        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_es_citrinet_512",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_es_citrinet_512",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_citrinet_512/versions/1.0.0/files/stt_es_citrinet_512.nemo",
        )

        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_small",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_small",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_small/versions/1.0.0/files/stt_en_conformer_ctc_small.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_medium",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_medium",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_medium/versions/1.0.0/files/stt_en_conformer_ctc_medium.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_large/versions/1.0.0/files/stt_en_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_small_ls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_small_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_small_ls/versions/1.0.0/files/stt_en_conformer_ctc_small_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_medium_ls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_medium_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_medium_ls/versions/1.0.0/files/stt_en_conformer_ctc_medium_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_large_ls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_large_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_large_ls/versions/1.0.0/files/stt_en_conformer_ctc_large_ls.nemo",
        )
        results.append(model)

        return results
