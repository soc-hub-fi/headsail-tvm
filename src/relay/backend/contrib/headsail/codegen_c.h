#ifndef TVM_RELAY_BACKEND_CONTRIB_HEADSAIL_CODEGEN_C_H_
#define TVM_RELAY_BACKEND_CONTRIB_HEADSAIL_CODEGEN_C_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace contrib {

struct Output {
  std::string name;
  std::string dtype;
  int size;
  bool need_copy;
};

struct GenerateBodyOutput {
  std::string decl;
  std::vector<std::string> buffers;
  std::vector<Output> outputs;
  Array<String> headers;
};

#endif // TVM_RELAY_BACKEND_CONTRIB_HEADSAIL_CODEGEN_C_H_
