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
#include <cstring>

#include "../../utils.h"
#include "./codegen_headsail.h"
//#include "../codegen_c/codegen_c.h"
#include "../../../../target/source/codegen_c_host.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

inline size_t GetShape1DSize(const Type& type) {
  const auto shape = GetShape(type);
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}


std::vector<std::string> Conv2d_bias(const CallNode* call) {

  std::vector<std::string> args;

  const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
  ICHECK(conv2d_attr);

  auto ishape = GetShape(call->args[0]->checked_type()); // Input shape
  auto wshape = GetShape(call->args[1]->checked_type()); // Kernel shape

  std::cout << std::endl;
  // args.push_back(std::to_string(ishape[0])); // Input batch
  // std::cout << "Input batch: " << std::to_string(ishape[0]) << std::endl;

  args.push_back(std::to_string(ishape[1])); // Input height
  std::cout << "Input height: " << std::to_string(ishape[1]) << std::endl;

  args.push_back(std::to_string(ishape[2])); // Input width
  std::cout << "Input width: " << std::to_string(ishape[2]) << std::endl;

  args.push_back(std::to_string(ishape[3])); // Input channels
  std::cout << "Input channel: " << std::to_string(ishape[3]) << std::endl;

  // Input layout
  char data_layout[6];
  std::strcpy(data_layout, "\"");
  std::strcat(data_layout, &conv2d_attr->data_layout.c_str()[1]);
  std::strcat(data_layout, "\"");

  std::cout << "Data layout: " << data_layout << std::endl;
  args.push_back(data_layout);

  std::cout << "Data layout: " << conv2d_attr->weight.c_str()<< std::endl;

  args.push_back(std::to_string(wshape[0])); // Kernels amount
  std::cout << "Kernels amount: " << std::to_string(wshape[0]) << std::endl;

  args.push_back(std::to_string(wshape[1])); // Kernels height
  std::cout << "Kernels height: " << std::to_string(wshape[1]) << std::endl;

  args.push_back(std::to_string(wshape[2])); // Kernels width
  std::cout << "Kernels width: " << std::to_string(wshape[2]) << std::endl;

  args.push_back(std::to_string(wshape[3])); // Kernels channels
  std::cout << "Kernels channels: " << std::to_string(wshape[3]) << std::endl;

  // Kernel layout
  char kernel_layout[7];
  std::strcpy(kernel_layout, "\"");
  std::strcat(kernel_layout, conv2d_attr->kernel_layout.c_str());
  std::strcat(kernel_layout, "\"");

  // Convert TVM layout string to Headsail layout string
  for (int i = 0; i < 7; ++i) {
    if (kernel_layout[i] == 'I') {
      kernel_layout[i] = 'K';
    } else if (kernel_layout[i] == 'O') {
      kernel_layout[i] = 'C';
    }
  }
  std::cout << "Kernel layout: " << kernel_layout << std::endl;
  args.push_back(kernel_layout);

  std::cout << "Bias size: " << std::to_string(conv2d_attr->groups * wshape[3]) << std::endl;
  args.push_back(std::to_string(conv2d_attr->groups * wshape[3]));

  // Padding
  args.push_back(std::to_string(conv2d_attr->padding[0].as<IntImmNode>()->value)); // Pad top
  args.push_back(std::to_string(conv2d_attr->padding[1].as<IntImmNode>()->value)); // Pad left
  args.push_back(std::to_string(conv2d_attr->padding[3].as<IntImmNode>()->value)); // Pad right
  args.push_back(std::to_string(conv2d_attr->padding[2].as<IntImmNode>()->value)); // Pad bottom
  args.push_back(std::to_string(0));                                               // Pad value

  args.push_back(std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value)); // Stride x
  args.push_back(std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value)); // Stride y
  args.push_back(std::to_string(0));                                               // Mac clip
  args.push_back(std::to_string(0));                                               // PP clip

  return args;

}


class CodegenHeadsail : public MemoizedExprTranslator<std::vector<Output>>, public HeadsailCodegenCBase {
    public:
        //CodegenHeadsail(const std::string& id) { this->ext_func_id_ = id; }
		CodegenHeadsail(std::unordered_map<std::string, runtime::NDArray>* const_name_to_constant,
           Array<String>* const_names, std::string ext_func_id)
      : const_name_to_constant_(const_name_to_constant),
        const_names_(const_names),
        ext_func_id_(std::move(ext_func_id)) {}

        std::vector<Output> VisitExprDefault_(const Object* op) final {
            LOG(FATAL) << "Headsail codegen doesn't support: " << op->GetTypeKey();
        }

        // Generates function parameter
        std::vector<Output> VisitExpr_(const VarNode* node) final {
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

			size_t const_id = const_name_to_constant_->size();

            // Get const: static_cast<float*>(dnnl_0_consts[0]->data)
            //output.name = CreateDataReference(ext_func_id_, const_idx_);
            output.name = CreateDataReference(ext_func_id_, const_id);
			const auto* type_node = cn->checked_type().as<TensorTypeNode>();
			ICHECK(type_node);
			const auto& dtype = GetDtypeString(type_node);

            // Generate the global variable for needed ndarrays
            if (const_array_name_.empty()) {
              const_array_name_ = CreateNDArrayPool(ext_func_id_);
              std::string checker = CreateInitChecker(ext_func_id_);
              ext_func_body_.insert(ext_func_body_.begin(), checker);
            }

			ICHECK(dtype == "int" || dtype == "int8_t") << "Only int8_t and int are supported for now.";
			output.dtype = dtype;

			std::string const_var_name = CreateConstVar(ext_func_id_, const_id);
			std::cout << "Data:"  << cn->data << std::endl;
			const_name_to_constant_->emplace(const_var_name, cn->data);
			const_names_->push_back(const_var_name);

			return {output};
            // ICHECK(GetDtypeString(type_node) == "int" || GetDtypeString(type_node) == "int8_t" );

            // // Give the ndarray a unique name to ease the initialization of it at
            // // runtime.
            // std::string const_symbol = "headsail_" + ext_func_id_;
            // std::string const_var_name = CreateConstVar(const_symbol, const_idx_);
            // const_vars_.push_back(const_var_name);
            // const_idx_++;

            // std::cout << "Const:" << const_symbol << std::endl;

            // const auto* type_node = cn->checked_type().as<TensorTypeNode>();
			// std::cout << "Type:" << GetDtypeString(type_node) << std::endl;

            // return {output};
        }

        std::vector<Output> VisitExpr_(const CallNode* call) final {
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

        std::string JIT(const std::vector<Output>& out) {
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
                //{"qnn.conv2d", {"dla_conv2d", Conv2d}},
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

            if (pattern_name == "headsail.tflite_conv2d_bias_relu") {
                const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 3,
                                                    (const std::vector<std::string>){"qnn.conv2d", "nn.bias_add", "qnn.requantize", "clip"});
                return GenerateBody(conv_call, "dla_conv2d_bias_relu", GetArgumentNames(caller),
                                    Conv2d_bias(conv_call));
            } else if (pattern_name == "headsail.tflite_conv2d_bias") {
                const auto* conv_call =
                    GetRootCall(callee->body.as<CallNode>(), 1, {"qnn.conv2d", "nn.bias_add", ""});
                return GenerateBody(conv_call, "dla_conv2d_bias", GetArgumentNames(caller),
                                    Conv2d_bias(conv_call));
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

            // Constant arguments i.e. input, weight, output
            decl_stream << "(" << func_args[0];
            for (size_t i = 1; i < func_args.size(); ++i) {
                decl_stream << ", " << func_args[i];
            }
            std::cout << "decl: " << decl_stream.str() << std::endl;

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

            // Attach attribute arguments, op specific defined by the codegen
            for (size_t i = 0; i < attribute_args.size(); ++i) {
                decl_stream << ", " << attribute_args[i];
				std::cout << "Arg:" << attribute_args[i] << std::endl;
            }
            decl_stream << ");";
            ret.decl = func_name + decl_stream.str();
            return ret;
        }

		/*!
		* \brief The accumulated constant name to constant mapping. Shared between all generated
		* functions.
		*/
		std::unordered_map<std::string, runtime::NDArray>* const_name_to_constant_;

        /*! \brief The id of the external dnnl ext_func. */
        std::string ext_func_id_;
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
        Array<String>* const_names_;

        friend class HeadsailModuleCodegen;
       };

class HeadsailModuleCodegen : public CSourceModuleCodegenBase {
 public:
  // Create a corresponding DNNL function for the given relay Function.
  std::pair<std::string, Array<String>> GenHeadsailFunc(const Function& func) {
    ICHECK(func.defined()) << "Input error: expect a Relay function.";

    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);
    func_names_.push_back(sid);

    CodegenHeadsail builder(&const_name_to_constant_, &const_names_, sid);
    auto out = builder.VisitExpr(func->body);
    code_stream_ << builder.JIT(out);

    return {sid, const_names_};
  }

    /*! \brief Returns the accumulated constant name to constant mapping. */
	const std::unordered_map<std::string, runtime::NDArray>& const_name_to_constant() const {
		return const_name_to_constant_;
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
    code_stream_ << "#include <stdio.h>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
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

    std::cout << "Sym: " << sym << std::endl;

    int i = 0;
    for (auto x : variables) {
        std::cout << i << " | " << "Var: " << x << std::endl;
        ++i;
    }

    // Create a CSource module
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    ICHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    //// TODO(@manupa-arm): pass the function names to enable system-lib creation
    //return (*pf)(code, "c", Array<String>{sym}, variables);
    // Use this if things break
    return codegen::CSourceModuleCreate(code, "c", func_names_);
  }

 private:
  /*!
   * \brief The code stream that prints the code that will be compiled using
   * external codegen tools.
   */
  std::ostringstream code_stream_;
  Array<String> func_names_;
  std::unordered_map<std::string, runtime::NDArray> const_name_to_constant_;
  Array<String> const_names_;
};


runtime::Module HeadsailCompiler(const ObjectRef& ref) {
    HeadsailModuleCodegen headsail;
    return headsail.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.headsail").set_body_typed(HeadsailCompiler);


}  // namespace contrib
}  // namespace relay
}  // namespace tvm
