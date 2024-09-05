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
from ...dataflow_pattern import DFPatternCallback, is_constant, is_expr, is_op, rewrite, wildcard
from tvm.relay.expr import Call, GlobalVar, TupleGetItem, const
from tvm.relay import transform
from .register import register_pattern_table

from ..strategy.generic import is_depthwise_conv2d
logger = logging.getLogger("HEADSAIL")

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


def qnn_tflite_conv2d_bias():
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    pattern = is_op("qnn.conv2d")(
             data, weight, is_constant(), is_constant(), is_constant(), is_constant()
    )
    #pattern = is_op("nn.bias_add")(pattern, bias)
    pattern = is_op("add")(pattern, bias)
    return pattern


@register_pattern_table("headsail")
def pattern_table():
    tflite_conv2d_bias= ("headsail.tflite_conv2d_bias", qnn_tflite_conv2d_bias())
    return [tflite_conv2d_bias]

class LegalizeQnnOpForHeadsail(DFPatternCallback):
    """Legalize QNN based patterns to match DNNL

    original pattern:
      OP = qnn.conv2d
      %1 = OP<int>(SRC, WGH) - OP<int>(src_zp, WGH)   // qnn.conv2d
      %2 = %1 + orig_bias                             // bias
      %2 = (%1 - rq_in_zp) * rq_in_scl / rq_out_scl + rq_out_zp  // qnn.requantize
      %3 = act(%2)                                               // activation == clip

    transform to Headsail compatible:
      %1 = OP<int>(SRC, WGH)
      %2 = (%1 + bias)
      %3 = cast(%2, dtype="float")
      %4 = act(%4) * act_scl
      %5 = %4 + SRC2 * sum_scl
      %6 = cast(%5, dtype="int8")

    where:
      act_scl = sum_lhs_scl / sum_out_scl
      sum_scl = sum_rhs_scl / sum_out_scl
    """

    def __init__(self):
        super(LegalizeQnnOpForHeadsail, self).__init__()
        self.src = wildcard()
        self.wgh = wildcard()
        self.bias = wildcard()
        self.sum_src = wildcard()

        self.src_scl = is_constant()
        self.src_zp = is_constant()
        self.wgh_scl = is_constant()
        self.wgh_zp = is_expr(const(0))

        self.rq_in_scl = is_constant()
        self.rq_in_zp = is_constant()
        self.rq_out_scl = is_constant()
        self.rq_out_zp = is_constant()

        self.sum_lhs_scl = is_constant()
        self.sum_lhs_zp = is_constant()
        self.sum_rhs_scl = is_constant()
        self.sum_rhs_zp = is_constant()
        self.sum_out_scl = is_constant()
        self.sum_out_zp = is_constant()


        self.root = (is_op("qnn.conv2d") | is_op("qnn.dense"))(
            self.src, self.wgh, self.src_zp, self.wgh_zp, self.src_scl, self.wgh_scl
        )
        pat = is_op("nn.bias_add")(self.root, self.bias) | self.root  # optional bias
        pat = is_op("qnn.requantize")(
            pat, self.rq_in_scl, self.rq_in_zp, self.rq_out_scl, self.rq_out_zp
        )
        self.clip = is_op("clip")(pat)
        pat = pat | self.clip

        add = is_op("qnn.add")(
            pat,
            self.sum_src,
            self.sum_lhs_scl,
            self.sum_lhs_zp,
            self.sum_rhs_scl,
            self.sum_rhs_zp,
            self.sum_out_scl,
            self.sum_out_zp,
        )
        add = is_op("clip")(add)
        self.pattern = pat | add


    def callback(self, pre, post, node_map):
        root = node_map[self.root][0]
        src = node_map[self.src][0]
        wgh = node_map[self.wgh][0]
        bias = node_map.get(self.bias, default=[relay.const(0, dtype="int32")])[0]
        src_scl = node_map[self.src_scl][0]
        src_zp = node_map[self.src_zp][0]
        rq_in_scl = node_map[self.rq_in_scl][0]
        rq_in_zp = node_map[self.rq_in_zp][0]
        rq_out_scl = node_map[self.rq_out_scl][0]
        rq_out_zp = node_map[self.rq_out_zp][0]
        final_dtype = "int8"

        print("src_scl", src_scl)
        print("src_zp",  src_zp)
        print("rq_in_scl", rq_in_scl)
        print("rq_in_zp", rq_in_zp)
        print("rq_out_scl", rq_out_scl)
        print("rq_out_zp", rq_out_zp)

        def cast_fp(op):
            return relay.op.cast(op, dtype="float32")
        def cast_int8(op):
            return relay.op.cast(op, dtype="int8")


        # Default values if qnn.sum is not present
        sum_src = node_map[self.sum_src][0] if self.sum_src in node_map else None
        sum_lhs_scl = node_map[self.sum_lhs_scl][0] if sum_src else relay.const(1, dtype="float32")
        sum_lhs_zp = node_map[self.sum_lhs_zp][0] if sum_src else relay.const(0, dtype="int32")
        sum_rhs_scl = node_map[self.sum_rhs_scl][0] if sum_src else relay.const(0, dtype="float32")
        sum_rhs_zp = node_map[self.sum_rhs_zp][0] if sum_src else relay.const(0, dtype="int32")
        sum_out_scl = node_map[self.sum_out_scl][0] if sum_src else relay.const(1, dtype="float32")
        sum_out_zp = node_map[self.sum_out_zp][0] if sum_src else relay.const(0, dtype="int32")


        # Compute scaling factors for requantization
        zero_zp = relay.const(0, dtype="int32")
        act_scl = sum_lhs_scl / sum_out_scl
        sum_scl = sum_rhs_scl / sum_out_scl

        # Remove zero-point
        rq_in_zp = zero_zp
        rq_out_zp = zero_zp

        # Construct the new computation graph
        output = tvm.relay.Call(
            root.op,
            [src, wgh, zero_zp, zero_zp, relay.const(1.0, dtype="float32"), relay.const(1.0, dtype="float32")],
            root.attrs,
            root.type_args,
            root.span,
        )
        output = output + bias

        # Insert requantize node back
        output = relay.qnn.op.requantize(
            output,
            input_scale=rq_in_scl,
            input_zero_point=rq_in_zp,
            output_scale=rq_out_scl,
            output_zero_point=rq_out_zp,
            out_dtype="int32"
        )

        # Apply clipping with optional ReLU
        if self.clip in node_map:
            output = relay.op.clip(output, 0, 127)
        else:
            output = relay.op.clip(output, -128, 127)

        # Apply qnn.add if sum was matched
        if sum_src:
            output = (cast_fp(output) * act_scl) + (cast_fp(sum_src) * sum_scl)
            output = relay.op.clip(output, 0, 127)

        # Cast to int8
        output = relay.op.cast(output, dtype=final_dtype)


        print("Legalization pass done")
        return output

def legalize_qnn_for_headsail(mod):
    """Transform qnn primitives to DNNL compatible form. Eliminate source zero point and apply
    strict sequence of post ops."""
    print("Legalizing qnn for headsail")
    mod["main"] = rewrite(LegalizeQnnOpForHeadsail(), mod["main"])

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            # transform.SimplifyInference(),  # TODO: this pass decompose nn.layer_norm
            # transform.FoldScaleAxis(),  # TODO: fail inside TVM in case of grouped convolutions.
            #transform.FoldConstant(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod
