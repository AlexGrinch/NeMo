#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../nemo"))

from package_info import __version__


autodoc_mock_imports = [
    'torch',
    'torch.nn',
    'torch.utils',
    'torch.optim',
    'torch.utils.data',
    'torch.utils.data.sampler',
    'torchvision',
    'torchvision.models',
    'torchtext',
    'torch_stft',
    'pytorch_lightning',
    'h5py',
    'kaldi_io',
    'transformers',
    'transformers.tokenization_bert',
    'transformers.BertModel',
    'apex',
    'ruamel',
    'frozendict',
    'inflect',
    'unidecode',
    'librosa',
    'soundfile',
    'sentencepiece',
    'youtokentome',
    'megatron-lm',
    'numpy',
    'dateutil',
    'wget',
    'scipy',
    'pandas',
    'matplotlib',
    'sklearn',
    'braceexpand',
    'webdataset',
    'tqdm',
    'numba',
    'hydra',
    'omegaconf',
    'onnx',
    'editdistance',
    'megatron',
    'pesq',
    'pystoi',
]

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinxcontrib.bibtex",
    "sphinx.ext.inheritance_diagram",
]

# Set default flags for all classes.
autodoc_default_options = {'members': None, 'undoc-members': None, 'show-inheritance': True}

locale_dirs = ['locale/']  # path is example but recommended.
gettext_compact = False  # optional.

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "NVIDIA NeMo"
copyright = "2018-, NVIDIA CORPORATION"
author = "NVIDIA CORPORATION"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.


# The short X.Y version.
# version = "0.10.0"
version = __version__
# The full version, including alpha/beta/rc tags.
# release = "0.9.0"
release = __version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"

# NVIDIA theme settings.
html_theme = 'nvidia_theme'

html_theme_path = ["."]

html_theme_options = {
    'display_version': True,
    'project_version': version,
    'project_name': project,
    'logo_path': None,
    'logo_only': True,
}
html_title = 'Introduction'

html_logo = html_theme_options["logo_path"]

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "nemodoc"
