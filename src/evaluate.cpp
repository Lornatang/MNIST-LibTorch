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

#include "../include/model.h"
#include "../include/utils.h"
#include <torch/cuda.h>
#include <torch/data.h>

auto main(int argc, const char *argv[]) -> int {
  if (argc < 2) {
    std::cerr << "Usage: ./eval <data-to-path>\n";
    std::cerr << "Example: ./detect ../data\n";
    return 0;
  }
  // Where to find the MNIST dataset.
  const char *data_root = argv[1];

  // The batch size for training.
  const int64_t batch_size = 64;

  std::vector<double> result;

  torch::manual_seed(1);

  // choice GPU or CPU
  torch::Device device = select_device();

  auto model = std::make_shared<Net>();
  torch::load(model, "../assets/model_best.pth");
  model->to(device);

  // Load dataset
  auto test_dataset =
      torch::data::datasets::MNIST(data_root,
                                   torch::data::datasets::MNIST::Mode::kTest)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), batch_size);

  result = evaluate(model, device, *test_loader, test_dataset_size);
}