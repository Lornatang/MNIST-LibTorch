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

#ifndef MODEL_H
#define MODEL_H
#include <torch/nn.h>

struct LeNet : torch::nn::Module {
  LeNet() {
    // Construct and register two Linear submodules.
    LeNet::fc1 = register_module("fc1", torch::nn::Linear(784, 64));
    LeNet::fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    LeNet::fc3 = register_module("fc3", torch::nn::Linear(32, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = x.view({x.size(0), 784});
    x = LeNet::fc1->forward(x);
    x = LeNet::fc2->forward(x);
    x = LeNet::fc3->forward(x);
    x = torch::log_softmax(x, /*dim=*/1);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

#endif // MODEL_H