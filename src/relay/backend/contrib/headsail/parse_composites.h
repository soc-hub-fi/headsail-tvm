#ifndef PARSE_COMPOSITES_H_
#define PARSE_COMPOSITES_H_

/*!
 * \brief Simple helper to accumulate composite function arguments and corresponding attributes
 * with indexes of them.
 */
class ArgPacker {
 public:
  ArgPacker(std::unordered_map<std::string, dmlc::any>* attrs, std::vector<tvm::relay::Expr>* args)
      : attrs_(attrs), args_(args) {}

  int Put(const tvm::relay::Expr& arg, std::string tag_name = "") {
    if (!arg.defined()) return -1;
    int idx = args_->size();
    args_->push_back(arg);
    if (!tag_name.empty()) {
      attrs_->operator[](tag_name) = dmlc_attr(idx);
    }
    return idx;
  }

 private:
  std::unordered_map<std::string, dmlc::any>* attrs_;
  std::vector<tvm::relay::Expr>* args_;
};


const tvm::relay::CallNode* ParseQnnConvComp(const tvm::relay::FunctionNode& comp_fn,
                                             std::unordered_map<std::string, dmlc::any>* ext_attrs,
                                             std::vector<tvm::relay::Expr>* args) {
  using namespace tvm::relay;

  // Pattern
  auto src = IsWildcard();
  auto wgh = IsWildcard();
  auto sum_src = IsWildcard();
  auto bias = IsConstant();

  auto o_scl = IsConstant();
  auto act_scl = IsConstant();
  auto sum_scl = IsConstant();
  auto dst_zp = IsConstant();

  DFPattern cnv;
  DFPattern pat;

  cnv = IsOp("qnn.conv2d")({src, wgh, IsConstant(), IsConstant(), IsConstant(), IsConstant()});
  pat = IsOp("cast")({cnv});
  pat = IsOp("add")({pat, bias}) || pat;
  pat = IsOp("multiply")({pat, o_scl});
  pat = IsOp("clip")({pat});
  pat = IsOp("multiply")({pat, act_scl}) || pat;
  pat = IsOp("add")({pat, sum_scl * IsOp("cast")({sum_src})}) || pat;
  pat = IsOp("add")({pat, dst_zp}) || pat;
  pat = IsOp("cast")({pat});

  // Check pattern match
  auto indexed_body = CreateIndexedGraph(comp_fn.body);
  DFPatternMatcher matcher(indexed_body.get());
  auto res = matcher.Match(pat, comp_fn.body);
  ICHECK(res) << "Mismatch of DNNL partitioner and codegen logic";

  // Handle arguments in deterministic order
  auto map = matcher.GetMemo();
  auto find = [&map](const DFPattern& pat) -> tvm::relay::Expr {
    if (map.count(pat)) return map.at(pat)[0];
    return {};
  };

  ArgPacker arg_holder(ext_attrs, args);
  arg_holder.Put(find(src));
  arg_holder.Put(find(wgh));
  arg_holder.Put(find(bias), "bias_idx");
  arg_holder.Put(find(sum_src), "sum_idx");
  arg_holder.Put(find(o_scl), "o_scl_idx");
  arg_holder.Put(find(act_scl), "act_scl_idx");
  arg_holder.Put(find(sum_scl), "sum_scl_idx");
  arg_holder.Put(find(dst_zp), "dst_zp_idx");

  // Activation. Default clip to simulate relu via uint8 cast
  std::vector<std::string> clip_attr{"clip"};
  auto act_scl_val = map.count(act_scl) ? find(act_scl) : constant(1.0);
  clip_attr.push_back(std::to_string(arg_holder.Put(act_scl_val)));      // act_scale
  clip_attr.push_back(std::to_string(arg_holder.Put(constant(0.0))));    // alpha
  clip_attr.push_back(std::to_string(arg_holder.Put(constant(255.0))));  // beta
  (*ext_attrs)["activation"] = dmlc_attr(clip_attr);

  return map.at(cnv)[0].as<CallNode>();
}

/*!
 * Parse composite function and return real args, additional attributes and root call node
 * @param comp_fn composite function to parse
 * @param ext_attrs attr collection with additional attributes
 * @param args real arguments of node
 * @return root call node
 */
const tvm::relay::CallNode* ParseComposite(const tvm::relay::FunctionNode& comp_fn,
                                           std::unordered_map<std::string, dmlc::any>* ext_attrs,
                                           std::vector<tvm::relay::Expr>* args) {
  auto comp = comp_fn.GetAttr<tvm::String>(tvm::relay::attr::kComposite);
  ICHECK(comp.defined()) << "DNNL JSON runtime only supports composite functions.";
  auto name = comp.value();

  const tvm::relay::CallNode* res = nullptr;
  if (name == "dnnl.qnn.conv2d")
    res = ParseQnnConvComp(comp_fn, ext_attrs, args);
  else if (name == "dnnl.qnn.dense")
    res = ParseQnnDenseComp(comp_fn, ext_attrs, args);
  return res;
}



#endif // PARSE_COMPOSITES_H_
