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
# pylint: disable=invalid-name, unused-argument
"""DNNL library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by DNNL.

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to DNNL.
"""
import logging
import tvm.ir
from tvm import relay
from ...dataflow_pattern import DFPatternCallback, wildcard, is_op, is_constant, is_expr
from tvm.relay.expr import Call, GlobalVar, TupleGetItem, const
from .register import register_pattern_table

from ..strategy.generic import is_depthwise_conv2d
logger = logging.getLogger("HEADSAIL")

conv2d_counter = True

def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by Headsail.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by Headsail.
    """
    @tvm.ir.register_op_attr(op_name, "target.headsail")
    def _func_wrapper(expr):
        args = expr.args
        typ = args[0].checked_type
        if typ.dtype != "int8":
            return False

        global conv2d_counter
        if conv2d_counter == True:
            conv2d_counter = False
        logger.info(expr.span)
        return supported

    return _func_wrapper


#_register_external_op_helper("qnn.add")
#_register_external_op_helper("qnn.conv2d")
#_register_external_op_helper("qnn.relu")

# Special case to handle tflite models converted to relay with fused activation
def qnn_tflite_conv2d_bias():
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    pattern = is_op("qnn.conv2d")(
             data, weight, is_constant(), is_constant(), is_constant(), is_constant()
    )
    pattern = is_op("nn.bias_add")(pattern, bias)
    return pattern

def qnn_conv2d_pattern():
    """Create pattern for qnn.conv2D with optional pad and/or optional fused relu."""
    conv2d_input = wildcard()
    qnn_conv2d = is_op("qnn.conv2d")(
        conv2d_input,
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )
    bias_add = is_op("nn.bias_add")(qnn_conv2d, is_constant())
    req = is_op("qnn.requantize")(
        qnn_conv2d | bias_add, is_constant(), is_constant(), is_constant(), is_constant()
    )
    clip_or_req = req.optional(is_op("clip"))
    return clip_or_req

def qnn_tflite_conv2d_bias_relu():
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    pattern = is_op("qnn.conv2d")(
             data, is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
    )
    pattern = is_op("nn.bias_add")(pattern, is_constant())
    pattern = is_op("qnn.requantize")(
          pattern, is_constant(), is_constant(), is_constant(), is_constant()
    )
    pattern = is_op("clip")(pattern)
    return pattern

@register_pattern_table("headsail")
def pattern_table():
    tflite_conv2d_bias_relu = ("headsail.tflite_conv2d_bias_relu", qnn_tflite_conv2d_bias_relu())
    #tflite_conv2d_bias_relu = ("headsail.tflite_conv2d_bias_relu", qnn_conv2d_pattern(), check_qnn_conv2d)
    #tflite_conv2d_bias= ("headsail.tflite_conv2d_bias", qnn_tflite_conv2d_bias())
    return [tflite_conv2d_bias_relu]
    #return [tflite_conv2d_bias_relu, tflite_conv2d_bias]
