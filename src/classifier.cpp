/*
 * Copyright 2020 Lorna Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ============================================================================
 */

#include "../include/utils.h"
#include <torch/script.h>

double classifier(cv::Mat &image, const std::shared_ptr<Net> &model,
                  torch::Device device) {
  // convert imag to tensor
  torch::Tensor image_tensor =
      torch::from_blob(image.data,
                       {1, image.rows, image.cols, image.channels()},
                       torch::kByte)
          .to(device);
  image_tensor = image_tensor.permute({0, 3, 1, 2}).to(device);
  image_tensor = image_tensor.toType(torch::kFloat).to(device);
  image_tensor = image_tensor.div(255).to(device);
  at::Tensor output = model->forward({image_tensor}).to(device);
  auto max_result = output.max(1, true);
  auto classes = std::get<1>(max_result).item<int>();
  return classes;
}