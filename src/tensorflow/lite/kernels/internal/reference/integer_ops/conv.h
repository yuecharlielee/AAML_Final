/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <stdio.h>

#include <algorithm>

#include "cfu.h"
#include "perf.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  perf_enable_counter(6);
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // printf("ConvPerChannel<%d>: batches=%ld, inH=%ld, inW=%ld, inD=%ld,
  // outH=%ld, outW=%ld, outD=%ld, fH=%ld, fW=%ld, fInD=%ld\n",
  //        static_cast<int>(sizeof(int32_t)), input_shape.Dims(0),
  //        input_shape.Dims(1), input_shape.Dims(2), input_shape.Dims(3),
  //        output_shape.Dims(1), output_shape.Dims(2), output_shape.Dims(3),
  //        filter_shape.Dims(1), filter_shape.Dims(2),
  //        filter_shape.Dims(3));

  // printf("  stride_width=%d, stride_height=%d\n", stride_width,
  // stride_height); printf("  dilation_width_factor=%d,
  // dilation_height_factor=%d\n",
  //        dilation_width_factor, dilation_height_factor);
  // printf("  pad_width=%d, pad_height=%d\n", pad_width, pad_height);

  // Set min and max value of the output.

  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  int M = output_depth;
  int N = output_height * output_width;
  // int K = filter_height * filter_width * filter_input_depth;

  // printf("M=%d, N=%d, K=%d\n", M, N, K);

  const int TILE_SIZE = 32;

  constexpr int MAX_K = 8192;
  int8_t im2col_buf[TILE_SIZE][MAX_K];

  // int filter_row_index[256000];
  // int filter_col_index[256000];
  // int8_t filter_val_arr[256000];
  // int filter_idx = 0;

  int32_t tile_acc[32][32];

  cfu_op0(6, input_offset, 0);  

  for (int batch = 0; batch < batches; ++batch) {
    for (int n_base = 0; n_base < N; n_base += TILE_SIZE) {
      int tile_width = std::min(TILE_SIZE, N - n_base);
      int group = 0;

      memset(im2col_buf, -input_offset, sizeof(im2col_buf));

      for (int j = 0; j < tile_width; ++j) {
        int n_curr = n_base + j;
        int out_y = n_curr / output_width;
        int out_x = n_curr % output_width;
        int in_y_origin = (out_y * stride_height) - pad_height;
        int in_x_origin = (out_x * stride_width) - pad_width;

        int k = 0;
        
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          int in_y = in_y_origin + dilation_height_factor * filter_y;
          for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
            int in_x = in_x_origin + dilation_width_factor * filter_x;
            bool is_point_inside_image = (in_x >= 0) && (in_x < input_width) &&
                                         (in_y >= 0) && (in_y < input_height);

            for (int in_channel = 0; in_channel < filter_input_depth;
                 ++in_channel) {
              if (is_point_inside_image) {
                im2col_buf[j][k] =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
              }

              cfu_op0(0, im2col_buf[j][k], (j << 16) | k);

              k++;
            }
          }
        }
      }

      // for (int p = 0; p < tile_width; ++p) {
      //   for (int c = 0; c < K; c++) {
      //     int8_t cfu_val = static_cast<int8_t>(cfu_op0(1, p, c) & 0xff);
      //     int8_t soft_val = im2col_buf[p][c];

      //     if (cfu_val != soft_val) {
      //       printf("Error at pixel=%d, channel=%d: expected %02x, got %02x\n",
      //              p, c, (unsigned char)soft_val, (unsigned char)cfu_val);
      //     }
      //   }
      // }

      for (int m_base = 0; m_base < M; m_base += TILE_SIZE) {
        memset(tile_acc, 0, sizeof(tile_acc));

        // filter_idx = 0;
        int current_group = m_base / filters_per_group;
        int group_start = current_group * filters_per_group;
        int group_end = group_start + filters_per_group;
        int max_tile_height_in_group = group_end - m_base;
        int tile_height = std::min(TILE_SIZE, M - m_base);
        tile_height = std::min(tile_height, max_tile_height_in_group);

        for (int i = 0; i < tile_height; ++i) {
          int out_channel = m_base + i;
          int k = 0;
          bool is_first_accumulation = true;


          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];

                if (filter_val != 0) {
                  // filter_row_index[filter_idx] = k;
                  // filter_col_index[filter_idx] = i;
                  // filter_val_arr[filter_idx] = static_cast<int8_t>(filter_val);
                  // filter_idx++;
                  if (is_first_accumulation) {
                      cfu_op0(5, filter_val, (i << 16) | k);
                      is_first_accumulation = false;
                  } else {
                      cfu_op0(3, filter_val, (i << 16) | k);
                  }
                }
                k++;
              }
            }
          }

          if (is_first_accumulation) {
              cfu_op0(5, 0, (i << 16) | 0); 
          }
        }

        // for (int i = 0; i < filter_idx; ++i) {
        //   for (int j = 0; j < tile_width; ++j) {
        //     tile_acc[filter_col_index[i]][j] +=
        //         filter_val_arr[i] * (im2col_buf[j][filter_row_index[i]]);
        //   }
        // }

        // printf("Software results\n");
        // for (int i = 0; i < tile_height; ++i) {
        //   for (int j = 0; j < tile_width; ++j) {
        //     printf("%02lx ", tile_acc[i][j]);
        //   }
        //   printf("\n");
        // }

        // printf("CFU results\n");
        // for (int i = 0; i < tile_height; ++i) {
        //   for (int j = 0; j < tile_width; ++j) {
        //     int32_t cfu_result = static_cast<int32_t>(cfu_op0(4, i, j));
        //     printf("%02lx ", cfu_result);
        //   }
        //   printf("\n");
        // }

        // for(int i = 0; i < tile_height; ++i) {
        //   for(int j = 0; j < tile_width; ++j) {
        //     int32_t cfu_result = static_cast<int32_t>(cfu_op0(4, i, j));
        //     if(cfu_result != tile_acc[i][j]) {
        //       printf("Mismatch at m=%d, n=%d: expected %ld, got %ld\n",
        //              m_base + i, n_base + j, tile_acc[i][j], cfu_result);
        //     }
        //   }
        // }

        for (int i = 0; i < tile_height; ++i) {
          for (int j = 0; j < tile_width; ++j) {
            int m_curr = m_base + i;
            int n_curr = n_base + j;
            int out_channel = m_curr;
            int out_y = n_curr / output_width;
            int out_x = n_curr % output_width;
            tile_acc[i][j] = static_cast<int32_t>(cfu_op0(4, i, j));


            if (bias_data) {
              tile_acc[i][j] += bias_data[out_channel];
            }

            tile_acc[i][j] = MultiplyByQuantizedMultiplier(
                tile_acc[i][j], output_multiplier[out_channel],
                output_shift[out_channel]);
            tile_acc[i][j] += output_offset;
            tile_acc[i][j] = std::max(tile_acc[i][j], output_activation_min);
            tile_acc[i][j] = std::min(tile_acc[i][j], output_activation_max);

            output_data[Offset(output_shape, batch, out_y, out_x,
                               out_channel)] =
                static_cast<int8_t>(tile_acc[i][j]);
          }
        }
      }
    }
  }
  perf_disable_counter(6);
}

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  // printf("ConvPerChannel<%d>: batches=%d, inH=%d, inW=%d, inD=%d, outH=%d,
  // outW=%d, outD=%d, fH=%d, fW=%d, fInD=%d\n",
  //        static_cast<int>(sizeof(AccumScalar)), batches, input_height,
  //        input_width, input_depth, output_height, output_width, output_depth,
  //        filter_height, filter_width, filter_input_depth);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_