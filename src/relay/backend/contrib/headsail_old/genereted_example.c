#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>

extern "C" void headsail_call_conv2d_int8(int8_t* headsail_0_in0, int8_t* out0) {
    int8* buf_0 = (int8_t*)std::malloc()

    //Call headsail-bsp conv2d layer
    buf_0 = conv2d(input, kernels, padding, stride);

}
