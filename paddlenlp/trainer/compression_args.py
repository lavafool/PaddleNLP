# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020-present the HuggingFace Inc. team.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
import types
from typing import List, Optional

import paddle

from ..utils.log import logger
from .training_args import TrainingArguments

__all__ = [
    "CompressionArguments",
]


@dataclass
class CompressionArguments(TrainingArguments):
    """
    CompressionArguments is the subset of the arguments we use in our example
    scripts **which relate to the training loop itself**.

    Using [`PdArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse)
    arguments that can be specified on the command line.

    Parameters:
        strategy (`str`):
            Compression strategy. It supports 'dynabert+ptq', 'dynabert' and 'ptq' now.
    """
    strategy: Optional[str] = field(
        default="dynabert+ptq",
        metadata={
            "help":
            "Compression strategy. It supports 'dynabert+ptq', 'dynabert' and 'ptq' now."
        },
    )
    # dynabert
    width_mult_list: Optional[List[float]] = field(
        default=None,
        metadata={
            "help":
            ("List of width multiplicator for pruning using DynaBERT strategy.")
        },
    )
    logging_steps: int = field(default=100,
                               metadata={"help": "Log every X updates steps."})

    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint every X updates steps."})

    warmup_ratio: float = field(
        default=0.1,
        metadata={
            "help": "Linear warmup over warmup_ratio fraction of total steps."
        })
    # ptq:
    algo_list: Optional[List[str]] = field(
        default=None,
        metadata={
        "help":
        "Algorithm list for Post-Quantization, and it supports 'hist', 'KL', " \
        "'mse', 'avg', 'abs_max' and 'emd'.'KL' uses KL-divergenc method to get " \
        "the KL threshold for quantized activations and get the abs_max value " \
        "forquantized weights. 'abs_max' gets the abs max value for activations " \
        "and weights. 'min_max' gets the min and max value for quantized " \
        "activations and weights. 'avg' gets the average value among the max " \
        "values for activations. 'hist' gets the value of 'hist_percent' " \
        "quantile as the threshold. 'mse' gets the value which makes the " \
        "quantization mse loss minimal."
    }, )

    batch_num_list: Optional[List[int]] = field(
        default=None,
        metadata={
        "help":
        "List of batch_num. 'batch_num' is the number of batchs for sampling. " \
        "the number of calibrate data is batch_size * batch_nums. " \
        "If batch_nums is None, use all data provided by data loader as calibrate data."
    }, )
    batch_size_list: Optional[List[int]] = field(
        default=None,
        metadata={
            "help":
            "List of batch_size. 'batch_size' is the batch of data loader."
        },
    )
    weight_quantize_type: Optional[str] = field(
        default='channel_wise_abs_max',
        metadata={
        "help":
        "Support 'abs_max' and 'channel_wise_abs_max'. This param only specifies " \
        "the fake ops in saving quantized model, and we save the scale obtained " \
        "by post training quantization in fake ops. Compared to 'abs_max', " \
        "the model accuracy is usually higher when it is 'channel_wise_abs_max'."
    }, )
    round_type: Optional[str] = field(
        default='round',
        metadata={
        "help":
        "The method of converting the quantized weights value float->int. " \
        "Currently supports ['round', 'adaround'] methods. Default is `round`, " \
        "which is rounding nearest to the integer. 'adaround' is refer to " \
        "https://arxiv.org/abs/2004.10568."
    }, )
    bias_correction: Optional[bool] = field(
        default=False,
        metadata={
            "help":
            "If set to True, use the bias correction method of " \
            "https://arxiv.org/abs/1810.05723. Default is False."
        },
    )
    input_infer_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "If you have only inference model, quantization is also supported." \
            " The format is `dirname/file_prefix` or `file_prefix`. Default " \
            "is None."
        },
    )

    def print_config(self, args=None, key=""):
        """
        Prints all config values.
        """

        compression_arg_name = [
            'width_mult_list', 'batch_num_list', 'bias_correction',
            'round_type', 'algo_list', 'batch_size_list', 'strategy',
            'weight_quantize_type', 'input_infer_model_path'
        ]
        default_arg_dict = {
            "width_mult_list": [0.75],
            'batch_size_list': [1],
            'algo_list': ['mse', 'KL'],
            'batch_num_list': [4, 8, 16]
        }
        logger.info("=" * 60)
        if args is None:
            args = self
            key = "Compression"

        logger.info('{:^40}'.format("{} Configuration Arguments".format(key)))
        if key == "Compression":
            logger.info("Compression Suggestions: `Strategy` supports 'dynabert+ptq'," \
            "'dynabert' and 'ptq'. `width_mult_list` is needed in " \
            "`dynabert`, and `algo_list`, `batch_num_list`, `batch_size_list`," \
            " `round_type`, `bias_correction`, `weight_quantize_type`, " \
            "`input_infer_model_path` are needed in 'ptq'. "
            )
        logger.info('{:30}:{}'.format("paddle commit id",
                                      paddle.version.commit))

        for arg in dir(args):
            if key == "Compression" and arg not in compression_arg_name:
                continue
            if arg[:2] != "__":  #don't print double underscore methods
                v = getattr(args, arg)
                if v is None and arg in default_arg_dict:
                    v = default_arg_dict[arg]
                    setattr(args, arg, v)
                if not isinstance(v, types.MethodType):
                    logger.info('{:30}:{}'.format(arg, v))

        logger.info("")
