// /* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================*/
// #ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_
// #define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_

// #include <algorithm>
// #include <limits>
// #include <stdio.h>
// #include "perf.h"
// #include <cstring>

// #include "tensorflow/lite/kernels/internal/common.h"
// #include "cfu.h"

// namespace tflite {
// namespace reference_ops {

// inline void LeakyRelu(const tflite::LeakyReluParams& params,
//                       const RuntimeShape& input_shape, const float* input_data,
//                       const RuntimeShape& output_shape, float* output_data) {
//   const int flat_size = MatchingFlatSize(input_shape, output_shape);
//   for (int i = 0; i < flat_size; ++i) {
//     const float val = input_data[i];
//     // Note that alpha might be > 1 or < 0, so we don't use std::max here.
//     output_data[i] = val > 0 ? val : val * params.alpha;
//   }

//   printf("LeakyRelu: alpha=%f\n", static_cast<double>(params.alpha));
// }

// template <typename T>
// inline void QuantizeLeakyRelu(const LeakyReluParams& params,
//                               const RuntimeShape& input_shape,
//                               const T* input_data,
//                               const RuntimeShape& output_shape,
//                               T* output_data) {
//   const int flat_size = MatchingFlatSize(input_shape, output_shape);
//   static const int32_t quantized_min = std::numeric_limits<T>::min();
//   static const int32_t quantized_max = std::numeric_limits<T>::max();
//   perf_enable_counter(5);                                
//   float alpha_value = params.alpha;
//   uint32_t alpha_bits;
//   std::memcpy(&alpha_bits, &alpha_value, sizeof(float));

//   cfu_op0(19, params.output_multiplier_identity, 0);  // LOAD_POS_MULTIPLIER
//   cfu_op0(20, params.output_multiplier_alpha, 0);     // LOAD_NEG_MULTIPLIER
//   cfu_op0(21, params.output_shift_identity, 0);       // LOAD_POS_SHIFT
//   cfu_op0(22, params.output_shift_alpha, 0);          // LOAD_NEG_SHIFT
//   cfu_op0(23, quantized_min, 0); // LOAD_OUTPUT_MIN (傳送 -128)
//   cfu_op0(24, quantized_max, 0); // LOAD_OUTPUT_MAX (傳送 127)
//   cfu_op0(26, params.input_offset, 0);                 // LOAD_INPUT_OFFSET
//   cfu_op0(27, params.output_offset, 0);                // LOAD_OUTPUT_OFFSET

//   // // // Read back and verify
//   // int32_t read_pos_mult = cfu_op0(28, 0, 0);   // READ_POS_MULTIPLIER
//   // int32_t read_neg_mult = cfu_op0(29, 0, 0);   // READ_NEG_MULTIPLIER
//   // int32_t read_pos_shift = cfu_op0(30, 0, 0);  // READ_POS_SHIFT
//   // int32_t read_neg_shift = cfu_op0(31, 0, 0);  // READ_NEG_SHIFT
//   // int32_t read_out_min = cfu_op0(32, 0, 0);    // READ_OUTPUT_MIN
//   // int32_t read_out_max = cfu_op0(33, 0, 0);    // READ_OUTPUT_MAX
//   // int32_t read_input_offset = cfu_op0(36, 0, 0); // READ_INPUT_OFFSET
//   // int32_t read_output_offset = cfu_op0(37, 0, 0); // READ_OUTPUT_OFFSET

//   // printf("  Output Multiplier Identity: Loaded=%ld, ReadBack=%ld\n",
//   //        params.output_multiplier_identity, read_pos_mult);
//   // printf("  Output Multiplier Alpha   : Loaded=%ld, ReadBack=%ld\n",
//   //        params.output_multiplier_alpha, read_neg_mult);
//   // printf("  Output Shift Identity     : Loaded=%ld, ReadBack=%ld\n",
//   //        params.output_shift_identity, read_pos_shift);
//   // printf("  Output Shift Alpha        : Loaded=%ld, ReadBack=%ld\n",
//   //        params.output_shift_alpha, read_neg_shift);
//   // printf("  Output Min                : Loaded=%ld, ReadBack=%ld\n",
//   //        std::numeric_limits<int32_t>::min(), read_out_min);
//   // printf("  Output Max                : Loaded=%ld, ReadBack=%ld\n",
//   //        std::numeric_limits<int32_t>::max(), read_out_max);
//   // printf("  Input Offset             : Loaded=%ld, ReadBack=%ld\n", params.input_offset, read_input_offset);
//   // printf("  Output Offset            : Loaded=%ld, ReadBack=%ld\n", params.output_offset, read_output_offset);

//   for (int i = 0; i < flat_size; ++i) {

//     // const int32_t input_value = input_data[i] - params.input_offset;
//     // int32_t unclamped_output;

//     // if (input_value >= 0) {
//     //   unclamped_output = params.output_offset +
//     //                      MultiplyByQuantizedMultiplier(
//     //                          input_value, params.output_multiplier_identity,
//     //                          params.output_shift_identity);
//     // } else {
//     //   unclamped_output = params.output_offset +
//     //                      MultiplyByQuantizedMultiplier(
//     //                          input_value, params.output_multiplier_alpha,
//     //                          params.output_shift_alpha);
//     // }
//     // const T clamped_output =
//     //     std::min(quantized_max, std::max(quantized_min, unclamped_output));
//     // output_data[i] = static_cast<T>(clamped_output);


//     int32_t input_val_for_cfu = input_data[i];

 
//     cfu_op0(25, input_val_for_cfu, 0); 

//     int32_t cfu_raw_result = cfu_op0(35, 0, 0);

//     output_data[i] = static_cast<T>(cfu_raw_result);

//     // if(cfu_raw_result != static_cast<int32_t>(clamped_output)){
//     //   printf("Index %d Mismatch:\n", i);
//     //   printf("  Input (Raw): %d\n", input_data[i]);
//     //   printf("  Input (Adj): %ld\n", input_val_for_cfu);
//     //   printf("  SW (Ref)   : %ld\n", static_cast<int32_t>(clamped_output));
//     //   printf("  CFU (Raw)  : %ld\n", cfu_raw_result);
//     //   printf("--------------------------------\n");
//     // }
//   }
//   perf_disable_counter(5);

// }  // namespace reference_ops
// }  // namespace tflite
// }

// #endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_


/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"

#include "cfu.h" // 記得引入 CFU 標頭檔

namespace tflite {
namespace reference_ops {

// Float 版本保持不變 (硬體只加速量化版本)
inline void LeakyRelu(const tflite::LeakyReluParams& params,
                      const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    output_data[i] = val > 0 ? val : val * params.alpha;
  }
}

// [修改] 量化版本：使用 CFU 硬體加速
template <typename T>
inline void QuantizeLeakyRelu(const LeakyReluParams& params,
                              const RuntimeShape& input_shape,
                              const T* input_data,
                              const RuntimeShape& output_shape,
                              T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  
  // 1. 設定硬體參數 (只需做一次)
  // Op 7: 設定正向參數 (Multiplier, Shift)
  cfu_op0(7, params.output_multiplier_identity, params.output_shift_identity);

  // Op 8: 設定負向參數 (Multiplier, Shift)
  cfu_op0(8, params.output_multiplier_alpha, params.output_shift_alpha);

  printf("output_multiplier_identity=%lx, output_shift_identity=%lx\n",
         params.output_multiplier_identity, params.output_shift_identity);

  printf("output_multiplier_alpha=%lx, output_shift_alpha=%lx\n",
         params.output_multiplier_alpha, params.output_shift_alpha);

  // Op 9: 設定 Offsets (Input Offset, Output Offset)
  cfu_op0(9, params.input_offset, params.output_offset);

  // Op 11: 設定 Clamping 範圍 (Min, Max)
  // 根據 T 的型別 (int8 或 int16) 取得對應的最大最小值
  static const int32_t quantized_min = std::numeric_limits<T>::min();
  static const int32_t quantized_max = std::numeric_limits<T>::max();
  cfu_op0(11, quantized_min, quantized_max);

  // 2. 執行加速迴圈
  for (int i = 0; i < flat_size; ++i) {
    // 讀取原始數據
    int32_t input_val = input_data[i];

    // [Op 10] 執行 Leaky ReLU 計算
    // 硬體會自動執行: (input - in_offset) * mult >> shift + out_offset -> clamp
    int32_t res = cfu_op0(10, input_val, 0);

    // 寫回結果
    output_data[i] = static_cast<T>(res);
  }
}

}  // namespace reference_ops
}  // namespace tflite


#endif