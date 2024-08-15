/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <sstream>
#include <numeric>

#include "../../utils.h"
#include "./codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

inline size_t GetShape1DSize(const Type& type) {
  const auto shape = GetShape(type);
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

inline std::string GetShapeString(std::vector<int> shape) {
  std::string v = "std::vector<long int>{";
  for (auto s : shape) {
    v += std::to_string(s) + ",";
  }
  v += "}";
  return v;
}


std::vector<std::string> Conv2d(const CallNode* call) {
  std::vector<std::string> args;
  const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
  ICHECK(conv2d_attr);

  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  // Args: O, G, Ph0, Pw0, Ph1, Pw1, Kh, Kw, Sh, Sw
  // args.push_back(std::to_string(wshape[0]));
  // args.push_back(std::to_string(conv2d_attr->groups));
  // args.push_back(std::to_string(conv2d_attr->padding[0].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->padding[1].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->padding[2].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->padding[3].as<IntImmNode>()->value));
  // args.push_back(std::to_string(wshape[2]));
  // args.push_back(std::to_string(wshape[3]));
  // args.push_back(std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value));

  return args;
}

std::vector<std::string> Conv2d_bias(const CallNode* call) {
  std::vector<std::string> args;
  const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
  ICHECK(conv2d_attr);

  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  // Args: O, G, Ph0, Pw0, Ph1, Pw1, Kh, Kw, Sh, Sw
  // args.push_back(std::to_string(wshape[0]));
  // args.push_back(std::to_string(conv2d_attr->groups));
  // args.push_back(std::to_string(conv2d_attr->padding[0].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->padding[1].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->padding[2].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->padding[3].as<IntImmNode>()->value));
  // args.push_back(std::to_string(wshape[2]));
  // args.push_back(std::to_string(wshape[3]));
  // args.push_back(std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value));

  return args;
}

std::vector<std::string> Conv2d_relu(const CallNode* call) {
  std::vector<std::string> args;
  const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
  ICHECK(conv2d_attr);

  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  // Args: O, G, Ph0, Pw0, Ph1, Pw1, Kh, Kw, Sh, Sw
  // args.push_back(std::to_string(wshape[0]));
  // args.push_back(std::to_string(conv2d_attr->groups));
  // args.push_back(std::to_string(conv2d_attr->padding[0].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->padding[1].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->padding[2].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->padding[3].as<IntImmNode>()->value));
  // args.push_back(std::to_string(wshape[2]));
  // args.push_back(std::to_string(wshape[3]));
  // args.push_back(std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value));

  return args;
}

std::vector<std::string> Conv2d_bias_relu(const CallNode* call) {
  std::vector<std::string> args;
  const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
  ICHECK(conv2d_attr);

  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  // Args: O, G, Ph0, Pw0, Ph1, Pw1, Kh, Kw, Sh, Sw
  // args.push_back(std::to_string(wshape[0]));
  // args.push_back(std::to_string(conv2d_attr->groups));
  // args.push_back(std::to_string(conv2d_attr->padding[0].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->padding[1].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->padding[2].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->padding[3].as<IntImmNode>()->value));
  // args.push_back(std::to_string(wshape[2]));
  // args.push_back(std::to_string(wshape[3]));
  // args.push_back(std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value));
  // args.push_back(std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value));

  return args;
}


std::vector<std::string> Relu(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());
  // Args: N, C, H, W
  for (int i = 0; i < 4; ++i) {
      if (i < ishape.size()) {
          args.push_back(std::to_string(ishape[i]));
      } else {
          args.push_back("1");
    }
  }
  return args;
}

std::vector<std::string> Add(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());
  args.push_back(std::to_string(0));
  // Args: H, W
  for (int i = 0; i < 2; ++i) {
      if (i < ishape.size()) {
          args.push_back(std::to_string(ishape[i]));
      } else {
          args.push_back("1");
    }
  }
  return args;
}

class CodegenHeadsail : public MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {

    public:
        explicit CodegenHeadsail(const std::string& id) { this->ext_func_id_ = id; }

        std::vector<Output> VisitExprDefault_(const Object* op) final {
            LOG(FATAL) << "Headsail codegen doesn't support: " << op->GetTypeKey();
        }

        // Generates function parameter
        std::vector<Output> VisitExpr_(const VarNode* node) override {
            ext_func_args_.push_back(GetRef<Var>(node));
            Output output;
            output.name = node->name_hint();
            return {output};
        }

        std::vector<Output> VisitExpr_(const TupleNode* node) final {
            std::vector<Output> outs;
            for (auto field : node->fields) {
            auto res = VisitExpr(field);
            ICHECK_EQ(res.size(), 1U) << "Do not support tuple nest";
            outs.push_back(res[0]);
            }
            return outs;
        }

        std::vector<Output> VisitExpr_(const TupleGetItemNode* op) final {
            auto res = VisitExpr(op->tuple);
            ICHECK_GT(res.size(), static_cast<size_t>(op->index));

            // Only keep the item we want for the child node.
            // FIXME(@comaniac): The other items should still be requried for the primary outputs.
            return {res[op->index]};
        }

        std::vector<Output> VisitExpr_(const ConstantNode* cn) final {
            Output output;
            // Get const: static_cast<float*>(dnnl_0_consts[0]->data)
            output.name = CreateDataReference(ext_func_id_, const_idx_);
            output.dtype = "int8_t";

            // Generate the global variable for needed ndarrays
            if (const_array_name_.empty()) {
            const_array_name_ = CreateNDArrayPool(ext_func_id_);
            std::string checker = CreateInitChecker(ext_func_id_);
            ext_func_body_.insert(ext_func_body_.begin(), checker);
            }

            // Give the ndarray a unique name to ease the initialization of it at
            // runtime.
            std::string const_symbol = "headsail_" + ext_func_id_;
            std::string const_var_name = CreateConstVar(const_symbol, const_idx_);
            const_vars_.push_back(const_var_name);
            const_idx_++;

            const auto* type_node = cn->checked_type().as<TensorTypeNode>();
            ICHECK(type_node);
            ICHECK_EQ(GetDtypeString(type_node), "int8") << "Only int8 is supported for now.";

            return {output};
        }

        std::vector<Output> VisitExpr_(const CallNode* call) override {
            GenerateBodyOutput ret;
            if (const auto* func = call->op.as<FunctionNode>()) {
                ret = GenerateCompositeFunctionCall(func, call);

            } else {
                ret = GenerateOpCall(call);
            }

            buf_decl_.insert(buf_decl_.end(), ret.buffers.begin(), ret.buffers.end());
            ext_func_body_.push_back(ret.decl);
            return ret.outputs;
        }

        std::string JIT(const std::vector<Output>& out) override {
            return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
        }


    private:
        std::vector<std::string> GetArgumentNames(const CallNode* call) {
            std::vector<std::string> arg_names;
            for (size_t i = 0; i < call->args.size(); ++i) {
                auto res = VisitExpr(call->args[i]);
                for (const auto& out : res) {
                    arg_names.push_back(out.name);
                }
          }
          return arg_names;
        }

        GenerateBodyOutput GenerateOpCall(const CallNode* call) {
            const auto* op_node = call->op.as<OpNode>();
            ICHECK(op_node) << "Expect OpNode, but got " << call->op->GetTypeKey();

            using ArgFunType = std::function<std::vector<std::string>(const CallNode*)>;
            static const std::map<std::string, std::pair<std::string, ArgFunType>> op_map = {
                {"qnn.conv2d", {"dla_conv2d", Conv2d}},
                {"qnn.relu", {"dla_relu", Relu}},
                {"qnn.add", {"dla_add", Add}},
                {"nn.conv2d", {"dla_conv2d", Conv2d}},
                {"nn.relu", {"dla_relu", Relu}},
                {"nn.add", {"dla_add", Add}},
                {"nn.bias_add", {"dla_bias_add", Add}},
            };

            const auto op_name = GetRef<Op>(op_node)->name;
            const auto iter = op_map.find(op_name);
            if (iter != op_map.end()) {
                return GenerateBody(call, iter->second.first, iter->second.second(call));
            }

            LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
        }

        GenerateBodyOutput GenerateCompositeFunctionCall(const FunctionNode* callee,
                                                        const CallNode* caller) {
            const auto pattern_name = callee->GetAttr<runtime::String>(attr::kComposite);
            ICHECK(pattern_name.defined()) << "Only functions with composite attribute supported";

            if (pattern_name == "headsail.conv2d_bias_relu") {
                const auto* conv_call =
                    GetRootCall(callee->body.as<CallNode>(), 2, {"qnn.conv2d", "qnn.add", "qnn.relu"});
                return GenerateBody(conv_call, "dla_conv2d_bias_relu", GetArgumentNames(caller),
                                    Conv2d_bias(conv_call));
            } else if (pattern_name == "headsail.conv2d_relu") {
                const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1,
                                                    (const std::vector<std::string>){"qnn.conv2d", "qnn.relu"});
                return GenerateBody(conv_call, "dla_conv2d_relu", GetArgumentNames(caller),
                                    Conv2d_relu(conv_call));
            } else if (pattern_name == "headsail.conv2d_bias") {
                const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1,
                                                    (const std::vector<std::string>){"qnn.conv2d", "qnn.bias_add"});
                return GenerateBody(conv_call, "dla_conv2d_bias", GetArgumentNames(caller),
                                    Conv2d_bias_relu(conv_call));
            }

            LOG(FATAL) << "Unknown composite function:" << pattern_name;
        }

        GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                                        const std::vector<std::string>& attribute_args) {
            return GenerateBody(root_call, func_name, GetArgumentNames(root_call), attribute_args);
        }

        GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                                        const std::vector<std::string>& func_args,
                                        const std::vector<std::string>& attribute_args) {
            // Make function call with input buffers when visiting arguments
            ICHECK_GT(func_args.size(), 0);
            std::ostringstream decl_stream;
            decl_stream << "(" << func_args[0];
            for (size_t i = 1; i < func_args.size(); ++i) {
                decl_stream << ", " << func_args[i];
            }

            // Analyze the output buffers
            std::vector<Type> out_types;
            if (root_call->checked_type()->IsInstance<TupleTypeNode>()) {
                auto type_node = root_call->checked_type().as<TupleTypeNode>();
                for (auto field : type_node->fields) {
                    ICHECK(field->IsInstance<TensorTypeNode>());
                    out_types.push_back(field);
                }
            } else if (root_call->checked_type()->IsInstance<TensorTypeNode>()) {
                ICHECK(root_call->checked_type()->IsInstance<TensorTypeNode>());
                out_types.push_back(root_call->checked_type());
            } else {
                LOG(FATAL) << "Unrecognized type node: " << AsText(root_call->checked_type(), false);
            }

            GenerateBodyOutput ret;
            for (const auto& out_type : out_types) {
                this->PrintIndents();
                const std::string out = "buf_" + std::to_string(buf_idx_++);
                const auto out_size = GetShape1DSize(out_type);
                decl_stream << ", " << out;

                Output output;
                output.name = out;
                output.size = out_size;
                output.dtype = GetDtypeString(out_type.as<TensorTypeNode>());
                output.need_copy = true;
                // NOTE: This needs to be extended for int16_t
                ret.buffers.push_back("int8_t* " + out + " = (int8_t*)malloc(" +
                                        std::to_string(out_size) + ");");
                ret.outputs.push_back(output);
            }

            // Attach attribute arguments
            for (size_t i = 0; i < attribute_args.size(); ++i) {
                decl_stream << ", " << attribute_args[i];
            }
            decl_stream << ");";
            ret.decl = func_name + decl_stream.str();
            return ret;
        }

        /*! \brief The id of the external dnnl ext_func. */
        std::string ext_func_id_{""};
        /*!
        * \brief The index to track the output buffer. Each kernel will redirect the
        * output to a buffer that may be consumed by other kernels.
        */
        int buf_idx_{0};
        /*! \brief The index of global constants. */
        int const_idx_{0};
        /*! \brief The arguments used by a wrapped function that calls DNNL kernels. */
        Array<Var> ext_func_args_;
        /*! \brief Statement of the function that will be compiled using DNNL kernels. */
        std::vector<std::string> ext_func_body_;
        /*! \brief The array declared to store the constant values. */
        std::string const_array_name_;
        /*! \brief The declaration of intermeidate buffers. */
        std::vector<std::string> buf_decl_;
        /*! \brief The variable name to constant mapping. */
        Array<String> const_vars_;

        friend class HeadsailModuleCodegen;
       };

class HeadsailModuleCodegen : public CSourceModuleCodegenBase {
 public:
  // Create a corresponding DNNL function for the given relay Function.
  std::pair<std::string, Array<String>> GenHeadsailFunc(const Function& func) {
    ICHECK(func.defined()) << "Input error: expect a Relay function.";

    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);

    CodegenHeadsail builder(sid);
    auto out = builder.VisitExpr(func->body);
    code_stream_ << builder.JIT(out);

    return {sid, builder.const_vars_};
  }

  /*!
   * \brief The overridden function that will create a CSourceModule. In order
   * to compile the generated C source code, users need to specify the paths to
   * some libraries, including some TVM required and dnnl specific ones. To make
   * linking simpiler, the DNNL kernels are wrapped in a TVM compatible manner
   * and live under tvm/src/runtime/contrib/dnnl folder.
   *
   * \param ref An object ref that could be either a Relay function or module.
   *
   * \return The runtime module that contains C source code.
   */
  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    // Create headers
    code_stream_ << "#include <stdint.h>\n";
    code_stream_ << "#include <stdlib.h>\n";
    code_stream_ << "#include <string.h>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <tvm/runtime/crt/packed_func.h>\n";
    code_stream_ << "#include <dlpack/dlpack.h>\n";
    // dnnl_kernel file is saved under src/runtime/contrib/dnnl so that we don't
    // expose it to ordinary users. To make export_library use it, users need to
    // pass -I${PATH_TO_TVM}/src/runtime/contrib
    code_stream_ << "#include <dla_driver.h>\n";
    code_stream_ << "\n";

    ICHECK(ref->IsInstance<FunctionNode>());
    auto res = GenHeadsailFunc(Downcast<Function>(ref));
    std::string code = code_stream_.str();
    String sym = std::get<0>(res);
    Array<String> variables = std::get<1>(res);

    // Create a CSource module
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    ICHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    // TODO(@manupa-arm): pass the function names to enable system-lib creation
    return (*pf)(code, "c", Array<String>{sym}, variables);
  }

 private:
  /*!
   * \brief The code stream that prints the code that will be compiled using
   * external codegen tools.
   */
  std::ostringstream code_stream_;
};


runtime::Module HeadsailCompiler(const ObjectRef& ref) {
    HeadsailModuleCodegen headsail;
    return headsail.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.headsail").set_body_typed(HeadsailCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
