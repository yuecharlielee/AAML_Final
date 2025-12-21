
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

  // printf("output_multiplier_identity=%lx, output_shift_identity=%lx\n",
  //        params.output_multiplier_identity, params.output_shift_identity);

  // printf("output_multiplier_alpha=%lx, output_shift_alpha=%lx\n",
  //        params.output_multiplier_alpha, params.output_shift_alpha);

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