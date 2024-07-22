# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=wildcard-import
"""Contrib modules."""
from .register import get_pattern_table, register_pattern_table

from .arm_compute_lib import *
from .dnnl import *
from .bnns import *
from .coreml import *
from .ethosn import *
from .libtorch import *
from .tensorrt import *
from .cutlass import *
from .clml import *
from .mrvl import *
from .headsail import *
