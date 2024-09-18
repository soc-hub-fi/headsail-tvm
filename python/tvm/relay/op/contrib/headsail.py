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
def qnn_tflite_conv2d_bias_relu():
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    pattern = is_op("qnn.conv2d")(
             #data, is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
             data, weight, is_constant(), is_constant(), is_constant(), is_constant()
    )
    #pattern = is_op("nn.bias_add")(pattern, is_constant())
    pattern = is_op("nn.bias_add")(pattern, bias)
    pattern = is_op("qnn.requantize")(
          pattern, is_constant(), is_constant(), is_constant(), is_constant()
    )
    pattern = is_op("clip")(pattern)
    return pattern

def make_qnn_conv2d_pattern():
    """Make qnn.conv2d based pattern supported by DNNL

    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    data = wildcard()
    weight = is_constant()
    bias = is_constant()
    o_scl = is_constant()
    dst_zp = is_constant()
    act_scl = is_constant()
    sum_scl = is_constant()
    sum_src = wildcard()

    zero_zp = is_expr(const(0, dtype="int32"))

    pat = is_op("qnn.conv2d")(data, weight, zero_zp, zero_zp, is_constant(), is_constant())
    pat = is_op("cast")(pat)
    pat = is_op("add")(pat, bias) | pat  # optional bias
    pat = is_op("multiply")(pat, o_scl)
    pat = is_op("clip")(pat)  # TBD, not only clip
    pat = is_op("multiply")(pat, act_scl) | pat  # optional multiply. Ex: act_scl == 1
    pat = is_op("add")(pat, sum_scl * is_op("cast")(sum_src)) | pat  # optional sum
    pat = is_op("add")(pat, dst_zp) | pat  # optional dst_zp, can be dst_zp == 0
    pat = is_op("cast")(pat)
    return pat

@register_pattern_table("headsail")
def pattern_table():
    tflite_conv2d_bias_relu = ("headsail.tflite_conv2d_bias_relu", qnn_tflite_conv2d_bias_relu())
    #tflite_conv2d_bias_relu = ("headsail.tflite_conv2d_bias_relu", make_qnn_conv2d_pattern())
    #tflite_conv2d_bias= ("headsail.tflite_conv2d_bias", qnn_tflite_conv2d_bias())
    return [tflite_conv2d_bias_relu]
    #return [tflite_conv2d_bias_relu, tflite_conv2d_b//ias]

class LegalizeQnnOpForHeadsail(DFPatternCallback):
    """Legalize QNN based patterns to match DNNL

    original pattern:
      OP = qnn.conv2d
      %1 = OP<int>(SRC, WGH) - OP<int>(src_zp, WGH)   // qnn.conv2d
      %2 = %1 + orig_bias                             // bias
      %2 = (%1 - rq_in_zp) * rq_in_scl / rq_out_scl + rq_out_zp  // qnn.requantize
      %3 = act(%2)                                               // activation == clip

    transform to DNNL compatible:
      %1 = OP<int>(SRC, WGH)
      %2 = cast(%1, dtype="float")
      %2 = (%1 + bias) * o_scl
      %3 = act(%2) * act_scl
      %4 = %3 + SRC2 * sum_scl
      %5 = %4 + dst_zp
      %6 = cast(%5, dtype="float")

    where:
      o_scl = rq_in_scl / rq_out_scl
      act_scl = sum_lhs_scl / sum_out_scl
      sum_scl = sum_rhs_scl / sum_out_scl
      bias = orig_bias - OP(src_zp, WGH) - rq_in_zp + rq_out_zp * rq_out_scl / rq_in_scl
      dst_zp = sum_out_zp - sum_lhs_zp * sum_lhs_scl / sum_out_scl -
               sum_rhs_zp * sum_rhs_scl / sum_out_scl
    """

    def __init__(self):
        super(LegalizeQnnOpForHeadsail, self).__init__()
        print("LEGALIZE _INIT_")
        self.src = wildcard()
        self.wgh = wildcard()
        self.bias = wildcard()

        self.src_scl = is_constant()
        self.src_zp = is_constant()
        self.wgh_scl = is_constant()
        self.wgh_zp = is_expr(const(0))

        self.rq_in_scl = is_constant()
        self.rq_in_zp = is_constant()
        self.rq_out_scl = is_constant()
        self.rq_out_zp = is_constant()

        self.root = is_op("qnn.conv2d")(
            self.src, self.wgh, self.src_zp, self.wgh_zp, self.src_scl, self.wgh_scl
        )
        pat = is_op("nn.bias_add")(self.root, self.bias) | self.root  # optional bias
        pat = is_op("qnn.requantize")(
            pat, self.rq_in_scl, self.rq_in_zp, self.rq_out_scl, self.rq_out_zp
        )
        pat = is_op("clip")(pat)
        self.pattern = pat

    def callback(self, pre, post, node_map):
        print("HERE!!!!!!!!!!!!!!!!!!!!")
        root = node_map[self.root][0]
        src = node_map[self.src][0]
        wgh = node_map[self.wgh][0]
        bias = node_map.get(self.bias, default=[relay.const(0, dtype="int32")])[0]
        src_zp = node_map[self.src_zp][0]
        input_scale = node_map[self.rq_in_scl][0]
        input_zp = node_map[self.rq_in_zp][0]
        out_scale = node_map[self.rq_out_scl][0]
        out_zp = node_map[self.rq_out_zp][0]

        print("src_zp", src_zp)
        print("input_scale", input_scale)
        print("input_zp", input_zp)
        print("out_scale", out_scale)
        print("out_zp", out_zp)



        #final_dtype = node_map[self.pattern][0].checked_type.dtype
        final_dtype = "int8"

        dst_layout = root.attrs.out_layout
        dst_layout = root.attrs.data_layout if dst_layout == "" else dst_layout
        wgh_layout = root.attrs.kernel_layout

        # TODO(@apeskov): dst_layout may ne blocked
        bias_rank = len(dst_layout) - dst_layout.index("C")


        def cast_fp(op):
            return relay.op.cast(op, dtype="float32")

        # recalculate some factors
        o_scl = input_scale / out_scale # Output scale
        #sum_scl = sum_rhs_scl / sum_out_scl
        # dst_zp = (
        #     cast_fp(sum_out_zp)
        #     - cast_fp(sum_lhs_zp) * sum_lhs_scl / sum_out_scl
        #     - cast_fp(sum_rhs_zp) * sum_rhs_scl / sum_out_scl
        # )
        bias = self.squeeze_bias(bias, dst_layout)
        bias = (
            cast_fp(bias)
            - cast_fp(self.fake_op(src_zp, wgh, wgh_layout))
            - cast_fp(input_zp)
            + cast_fp(out_zp) * out_scale / input_scale
        )
        bias = self.broadcast_to_rank(bias, bias_rank)

        zero_zp = relay.const(0, dtype="int32")
        one_scl = relay.const(1.0, dtype="float32")

        # construct new graph with proper post op ordering
        gr = tvm.relay.Call(
            root.op,
            [src, wgh, zero_zp, zero_zp, one_scl, one_scl],
            root.attrs,
            root.type_args,
            root.span,
        )
        #gr = relay.op.cast(gr, dtype="float32")
        gr = gr + bias
        gr = gr * o_scl
        #gr = relay.op.clip(gr, 0, 255) * act_scl
        #gr = gr + sum_scl * cast_fp(sum_src) if sum_src else gr
        #gr = gr + dst_zp
        gr = relay.op.cast(gr, dtype=final_dtype)
        return gr

    @staticmethod
    def fake_op(zp, wgh, layout):
        """Fake operator implementation for zp broadcast input"""
        # Conv:  reduce kernel {OC, IC, KH, KW} -> {OC} in case of group that is still correct
        # Dense: reduce kernel {OC, IC} -> {OC}
        wgh_int = relay.op.cast(wgh, dtype="int32")
        reduced_kernel = relay.op.sum(
            wgh_int, axis=[layout.index("O")], keepdims=False, exclude=True
        )
        return zp * reduced_kernel

    @staticmethod
    def squeeze_bias(bias, layout):
        shape = transform.InferTypeLocal(bias).concrete_shape
        c_position = layout.index("C") - len(layout) + len(shape)
        squeeze_idxs = [i for i in range(len(shape)) if i != c_position]
        return relay.op.squeeze(bias, squeeze_idxs)

    @staticmethod
    def broadcast_to_rank(op, rank):
        """Scalar or 1D tensor are supported"""
        shape = transform.InferTypeLocal(op).concrete_shape
        if len(shape) == 0:
            return op
        if len(shape) == 1:
            return relay.op.expand_dims(op, 1, rank - 1)
        raise ValueError("Unexpected bias rank to broadcast. Only 0 and 1 are supported.")

def legalize_qnn_for_headsail(mod):
    """Transform qnn primitives to DNNL compatible form. Eliminate source zero point and apply
    strict sequence of post ops."""
    print("Legalizing qnn for headsail")
    #mod["main"] = rewrite(LegalizeQnnOpForHeadsail(), mod["main"])

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            # transform.SimplifyInference(),  # TODO: this pass decompose nn.layer_norm
            # transform.FoldScaleAxis(),  # TODO: fail inside TVM in case of grouped convolutions.
            transform.FoldConstant(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod
