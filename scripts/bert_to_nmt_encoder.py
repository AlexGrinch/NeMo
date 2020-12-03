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

import argparse


parser = argparse.ArgumentParser(description='Converting BERT into NeMo NMT encoder')
parser.add_argument("--nemo_ckpt_path", required=True, default=None, type=str)
parser.add_argument("--bert_ckpt_path", required=True, default=None, type=str)
args = parser.parse_args()


import torch


def key2bert(nemonmt_key):
    key_components = nemonmt_key.split(".")

    if key_components[0] == "src_embedding_layer":
        key_components[0] = "embeddings"
        if key_components[1] == "token_embedding":
            key_components[1] = "word_embeddings"
        if key_components[1] == "position_embedding":
            key_components[1] = "position_embeddings"
            key_components[2] = "weight"
        if key_components[1] == "token_type_embedding":
            key_components[1] = "token_type_embeddings"
        if key_components[1] == "layer_norm":
            key_components[1] = "LayerNorm"

        return ".".join(key_components)

    elif key_components[0] == "encoder":
        key_components[1] = "layer"
        if key_components[3] == "first_sub_layer":
            key_components[3] = "attention"
            if key_components[4] == "query_net":
                key_components[4] = "self.query"
            if key_components[4] == "key_net":
                key_components[4] = "self.key"
            if key_components[4] == "value_net":
                key_components[4] = "self.value"
            if key_components[4] == "out_projection":
                key_components[4] = "output.dense"

        if key_components[3] == "second_sub_layer":
            if key_components[4] == "dense_in":
                key_components[3] = "intermediate"
                key_components[4] = "dense"
            if key_components[4] == "dense_out":
                key_components[3] = "output"
                key_components[4] = "dense"

        if key_components[3] == "layer_norm_1":
            key_components[3] = "attention.output.LayerNorm"

        if key_components[3] == "layer_norm_2":
            key_components[3] = "output.LayerNorm"

        return ".".join(key_components)

    else:
        return None


def bert_to_nemonmt(bert_state_dict, nemonmt_dict):
    
    for key in nemonmt_dict["state_dict"].keys():
        bert_key = key2bert(key)
        if bert_key is not None:
            nemonmt_dict["state_dict"][key] = bert_state_dict[bert_key]
    return nemonmt_dict


def main():
    nemonmt_dict = torch.load(args.nemo_ckpt_path)
    bert_state_dict = torch.load(args.bert_ckpt_path)
    nemonmt_dict = bert_to_nemonmt(bert_state_dict, nemonmt_dict)
    torch.save(nemonmt_dict, args.nemo_ckpt_path)


if __name__ == "__main__":
    main()
