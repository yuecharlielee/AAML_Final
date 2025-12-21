
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"

#include "cfu.h" 

namespace tflite {
namespace reference_ops {

inline void LeakyRelu(const tflite::LeakyReluParams& params,
                      const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    output_data[i] = val > 0 ? val : val * params.alpha;
  }
}

template <typename T>
inline void QuantizeLeakyRelu(const LeakyReluParams& params,
                              const RuntimeShape& input_shape,
                              const T* input_data,
                              const RuntimeShape& output_shape,
                              T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  
  cfu_op0(7, params.output_multiplier_identity, params.output_shift_identity);

  cfu_op0(8, params.output_multiplier_alpha, params.output_shift_alpha);

  // printf("output_multiplier_identity=%lx, output_shift_identity=%lx\n",
  //        params.output_multiplier_identity, params.output_shift_identity);

  // printf("output_multiplier_alpha=%lx, output_shift_alpha=%lx\n",
  //        params.output_multiplier_alpha, params.output_shift_alpha);

  cfu_op0(9, params.input_offset, params.output_offset);

  static const int32_t quantized_min = std::numeric_limits<T>::min();
  static const int32_t quantized_max = std::numeric_limits<T>::max();
  cfu_op0(11, quantized_min, quantized_max);

  for (int i = 0; i < flat_size; ++i) {

    int32_t input_val = input_data[i];

    int32_t res = cfu_op0(10, input_val, 0);

    output_data[i] = static_cast<T>(res);
  }
}

}  // namespace reference_ops
}  // namespace tflite


#endif