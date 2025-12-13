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

#include <algorithm>
#include <stdio.h>

#include "perf.h"
#include "cfu.h"
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

  // printf("ConvPerChannel<%d>: batches=%ld, inH=%ld, inW=%ld, inD=%ld, outH=%ld, outW=%ld, outD=%ld, fH=%ld, fW=%ld, fInD=%ld\n",
  //        static_cast<int>(sizeof(int32_t)), input_shape.Dims(0),
  //        input_shape.Dims(1), input_shape.Dims(2), input_shape.Dims(3),
  //        output_shape.Dims(1), output_shape.Dims(2), output_shape.Dims(3),
  //        filter_shape.Dims(1), filter_shape.Dims(2),
  //        filter_shape.Dims(3));
  
  // printf("  stride_width=%d, stride_height=%d\n", stride_width, stride_height);
  // printf("  dilation_width_factor=%d, dilation_height_factor=%d\n",
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
  // const int filters_per_group = output_depth;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  int M = output_depth; 
  int N = output_height * output_width; 
  int K = filter_height * filter_width * filter_input_depth;


  const int TILE_SIZE = 32; 
  int32_t tile_acc[32][32]; 


  printf("Starting Zero-Buffer GEMM M=%d, N=%d, K=%d\n", M, N, K);
  printf("batches=%d\n", batches);
  printf("offset=%ld\n", input_offset);


  for (int batch = 0; batch < batches; ++batch) {
    for (int m_base = 0; m_base < M; m_base += TILE_SIZE) {
    int tile_height = std::min(TILE_SIZE, M - m_base);
    int kernel_index = 0;


    for (int i = 0; i < tile_height; i += 4) { 
      for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
          for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
            
            int32_t packed_val = 0;

            for (int b = 0; b < 4; ++b) {
              int current_m_offset = i + b;
              int out_channel = m_base + current_m_offset;
              int8_t val = 0;

              if (current_m_offset < tile_height && out_channel < output_depth) {
                 val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
              }

              packed_val |= (static_cast<int32_t>(static_cast<uint8_t>(val)) << ((3 - b) * 8));
            }
            // printf("Kernel Packed Val[%d]: 0x%08lx\n", kernel_index, static_cast<uint32_t>(packed_val));

            cfu_op0(0, packed_val, kernel_index);
            kernel_index++;
          }
        }
      }
    }

    for (int n_base = 0; n_base < N; n_base += TILE_SIZE) {
      int tile_width = std::min(TILE_SIZE, N - n_base);


      int input_index = 0;


      for (int j = 0; j < tile_width; j += 4) {

        int in_y_origin[4];
        int in_x_origin[4];
        bool valid_col[4];

        for(int b = 0; b < 4; ++b) {
           int current_n_offset = j + b;
           if (current_n_offset < tile_width) {
               int n_curr = n_base + current_n_offset;
               int out_y = n_curr / output_width;
               int out_x = n_curr % output_width;

               in_y_origin[b] = (out_y * stride_height) - pad_height;
               in_x_origin[b] = (out_x * stride_width) - pad_width;
               valid_col[b] = true;
           } else {
               valid_col[b] = false;
               in_y_origin[b] = 0;
               in_x_origin[b] = 0;
           }
        }

        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
            for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
              
              int32_t packed_val = 0;
              for (int b = 0; b < 4; ++b) {
                int8_t val = static_cast<int8_t>(-input_offset);

                if (valid_col[b]) {
                  const int in_y = in_y_origin[b] + dilation_height_factor * filter_y;
                  const int in_x = in_x_origin[b] + dilation_width_factor * filter_x;
                  
                  const bool is_point_inside_image =
                      (in_x >= 0) && (in_x < input_width) && 
                      (in_y >= 0) && (in_y < input_height);

                  if (is_point_inside_image) {
                     val = input_data[Offset(input_shape, batch, in_y, in_x, in_channel)];
                  }
                }

                packed_val |= (static_cast<int32_t>(static_cast<uint8_t>(val)) << ((3 - b) * 8));
              }
              // printf("Input Packed Val[%d]: 0x%08lx\n", input_index, static_cast<uint32_t>(packed_val));
              cfu_op0(1, packed_val, input_index);
              input_index++;
            }
          }
        }
      }

      cfu_op0(2, K, tile_height << 16 | tile_width); 
      cfu_op0(11, input_offset, 0);
      cfu_op0(10, 0, 0);

      for(int idx = 0; idx < 32; idx++){
        for(int i = 0; i < 8; i++){
        
            int addr = (i * tile_height) + idx;
        
            tile_acc[idx][i*4+0] = cfu_op0(5, 0, addr);
            tile_acc[idx][i*4+1] = cfu_op0(6, 0, addr);
            tile_acc[idx][i*4+2] = cfu_op0(7, 0, addr);
            tile_acc[idx][i*4+3] = cfu_op0(8, 0, addr);
          
        }
     }



      

      for (int i = 0; i < tile_height; ++i) {
        for (int j = 0; j < tile_width; ++j) {

          int m_curr = m_base + i;
          int n_curr = n_base + j;

          int32_t acc = tile_acc[i][j];
          
          
          // printf("%08lx ", acc);

          if (bias_data) acc += bias_data[m_curr];

          
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[m_curr], output_shift[m_curr]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);

          int out_y = n_curr / output_width;
          int out_x = n_curr % output_width;
          output_data[Offset(output_shape, batch, out_y, out_x, m_curr)] = static_cast<int8_t>(acc);
        }
        // printf("\n");
      }
      
    } 
    }  // End of m_base loop
    
  }  // End of batch loop

  // //Print CFU output results after all batches are processed
  // printf("\n=== CFU Output Matrix ===\n");
  // for (int batch = 0; batch < batches; ++batch) {
  //   printf("Batch %d:\n", batch);
  //   for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
  //     for (int out_y = 0; out_y < output_height; ++out_y) {
  //       for (int out_x = 0; out_x < output_width; ++out_x) {
  //         int8_t val = output_data[Offset(output_shape, batch, out_y, out_x, out_channel)];
  //         printf("%02x ", val);
  //       }
        
  //     }
  //     printf("\n");
  //   }
  // }

  // // Check for indices exceeding 14-bit address space
  // printf("\n=== 14-BIT ADDRESS SPACE CHECK ===\n");
  // const int MAX_14BIT_ADDR = 16384; // 2^14
  // bool has_overflow = false;
  
  // for (int batch = 0; batch < batches; ++batch) {
  //   for (int m_curr = 0; m_curr < M; ++m_curr) {
  //     for (int n_curr = 0; n_curr < N; ++n_curr) {
  //       int out_y = n_curr / output_width;
  //       int out_x = n_curr % output_width;
        
  //       // Check if indices would overflow 14-bit space
  //       if (m_curr >= MAX_14BIT_ADDR || n_curr >= MAX_14BIT_ADDR) {
  //         printf("Batch %d, M[%d] N[%d] (out_y=%d, out_x=%d): INDEX OVERFLOW\n",
  //                batch, m_curr, n_curr, out_y, out_x);
  //         has_overflow = true;
  //       }
  //     }
  //   }
  // }
  
  // if (!has_overflow) {
  //   printf("All indices within 14-bit address space ✓\n");
  // }

  // // // Software GEMM verification - compute output for all channels
  // printf("\n=== COMPARISON: CFU vs SW Output (By Tile) ===\n");
  
  // int total_mismatches = 0;
  // bool batch_printed = false;
  
  // for (int batch = 0; batch < batches; ++batch) {
  //   // Iterate by tile
  //   for (int m_base = 0; m_base < M; m_base += TILE_SIZE) {
  //     int tile_height = std::min(TILE_SIZE, M - m_base);
      
  //     for (int n_base = 0; n_base < N; n_base += TILE_SIZE) {
  //       int tile_width = std::min(TILE_SIZE, N - n_base);
        
  //       // First pass: check for mismatches in this tile
  //       int tile_mismatches = 0;
  //       for (int i = 0; i < tile_height; ++i) {
  //         for (int j = 0; j < tile_width; ++j) {
  //           int m_curr = m_base + i;
  //           int n_curr = n_base + j;
  //           int out_channel = m_curr;
  //           int out_y = n_curr / output_width;
  //           int out_x = n_curr % output_width;
            
  //           // Get CFU output
  //           int8_t cfu_result = output_data[Offset(output_shape, batch, out_y, out_x, out_channel)];
            
  //           // Compute SW output
  //           auto group = out_channel / filters_per_group;
  //           int32_t acc = 0;
            
  //           const int in_y_origin = (out_y * stride_height) - pad_height;
  //           const int in_x_origin = (out_x * stride_width) - pad_width;
            
  //           for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
  //             const int in_y = in_y_origin + dilation_height_factor * filter_y;
  //             for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
  //               const int in_x = in_x_origin + dilation_width_factor * filter_x;

  //               const bool is_point_inside_image =
  //                   (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
  //                   (in_y < input_height);

  //               for (int in_channel = 0; in_channel < filter_input_depth;
  //                    ++in_channel) {
  //                 if (is_point_inside_image) {
  //                   int32_t input_val =
  //                       input_data[Offset(input_shape, batch, in_y, in_x,
  //                                         in_channel + group * filter_input_depth)];
  //                   int32_t filter_val = filter_data[Offset(
  //                       filter_shape, out_channel, filter_y, filter_x, in_channel)];
  //                   acc += filter_val * (input_val + input_offset);
  //                 }
  //               }
  //             }
  //           }

  //           if (bias_data) {
  //             acc += bias_data[out_channel];
  //           }
            
  //           acc = MultiplyByQuantizedMultiplier(
  //               acc, output_multiplier[out_channel], output_shift[out_channel]);
  //           acc += output_offset;
  //           acc = std::max(acc, output_activation_min);
  //           acc = std::min(acc, output_activation_max);
            
  //           int8_t sw_result = static_cast<int8_t>(acc);
            
  //           if (cfu_result != sw_result) {
  //             tile_mismatches++;
  //             total_mismatches++;
  //           }
  //         }
  //       }
        
  //       // Only print tile if it has mismatches
  //       if (tile_mismatches > 0) {
  //         if (!batch_printed) {
  //           printf("Batch %d:\n", batch);
  //           batch_printed = true;
  //         }
          
  //         printf("  Tile M[%d:%d] x N[%d:%d] (mismatches: %d):\n", m_base, m_base + tile_height - 1, 
  //                n_base, n_base + tile_width - 1, tile_mismatches);
          
  //         // Print SW results for this tile
  //         printf("    SW Output:\n");
  //         for (int i = 0; i < tile_height; ++i) {
  //           printf("      ");
  //           for (int j = 0; j < tile_width; ++j) {
  //             int m_curr = m_base + i;
  //             int n_curr = n_base + j;
  //             int out_channel = m_curr;
  //             int out_y = n_curr / output_width;
  //             int out_x = n_curr % output_width;
              
  //             auto group = out_channel / filters_per_group;
  //             int32_t acc = 0;
              
  //             const int in_y_origin = (out_y * stride_height) - pad_height;
  //             const int in_x_origin = (out_x * stride_width) - pad_width;
              
  //             for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
  //               const int in_y = in_y_origin + dilation_height_factor * filter_y;
  //               for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
  //                 const int in_x = in_x_origin + dilation_width_factor * filter_x;

  //                 const bool is_point_inside_image =
  //                     (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
  //                     (in_y < input_height);

  //                 for (int in_channel = 0; in_channel < filter_input_depth;
  //                      ++in_channel) {
  //                   if (is_point_inside_image) {
  //                     int32_t input_val =
  //                         input_data[Offset(input_shape, batch, in_y, in_x,
  //                                           in_channel + group * filter_input_depth)];
  //                     int32_t filter_val = filter_data[Offset(
  //                         filter_shape, out_channel, filter_y, filter_x, in_channel)];
  //                     acc += filter_val * (input_val + input_offset);
  //                   }
  //                 }
  //               }
  //             }

  //             if (bias_data) {
  //               acc += bias_data[out_channel];
  //             }
              
  //             acc = MultiplyByQuantizedMultiplier(
  //                 acc, output_multiplier[out_channel], output_shift[out_channel]);
  //             acc += output_offset;
  //             acc = std::max(acc, output_activation_min);
  //             acc = std::min(acc, output_activation_max);
              
  //             printf("%02x ", (uint8_t)acc);
  //           }
  //           printf("\n");
  //         }
          
  //         // Print CFU results for this tile
  //         printf("    CFU Output:\n");
  //         for (int i = 0; i < tile_height; ++i) {
  //           printf("      ");
  //           for (int j = 0; j < tile_width; ++j) {
  //             int m_curr = m_base + i;
  //             int n_curr = n_base + j;
  //             int out_channel = m_curr;
  //             int out_y = n_curr / output_width;
  //             int out_x = n_curr % output_width;
              
  //             int8_t cfu_result = output_data[Offset(output_shape, batch, out_y, out_x, out_channel)];
  //             printf("%02x ", (uint8_t)cfu_result);
  //           }
  //           printf("\n");
  //         }
  //         printf("\n");
  //       }
  //     }
  //   }
  //   batch_printed = false;
  // }
  
  // if (total_mismatches == 0) {
  //   printf("All outputs match! ✓\n");
  // } else {
  //   printf("Total mismatches: %d\n", total_mismatches);
  // }



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

  // printf("ConvPerChannel<%d>: batches=%d, inH=%d, inW=%d, inD=%d, outH=%d, outW=%d, outD=%d, fH=%d, fW=%d, fInD=%d\n",
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
