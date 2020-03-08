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

struct Net : torch::nn::Module {
  Net() {
    Net::conv1 = register_module("conv1", torch::nn::Conv2d(1, 10, 5));
    Net::maxpool1 = register_module("maxpool1", torch::nn::MaxPool2d(2));
    Net::relu1 = register_module("relu1", torch::nn::ReLU());
    Net::conv2 = register_module("conv2", torch::nn::Conv2d(10, 20, 5));
    Net::maxpool2 = register_module("maxpool2", torch::nn::MaxPool2d(2));
    Net::relu2 = register_module("relu2", torch::nn::ReLU());
    Net::fc1 = register_module("fc1", torch::nn::Linear(320, 128));
    Net::fc2 = register_module("fc2", torch::nn::Linear(128, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = Net::conv1(x);
    x = Net::maxpool1(x);
    x = Net::relu1(x);

    x = Net::conv2(x);
    x = Net::maxpool2(x);
    x = Net::relu2(x);

    x = x.view({-1, 320});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    x = torch::log_softmax(fc2->forward(x), /*dim=*/1);
    return x;
  }

  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::MaxPool2d maxpool1{nullptr}, maxpool2{nullptr};
  torch::nn::ReLU relu1{nullptr}, relu2{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

struct LeNet : torch::nn::Module {
  LeNet() {
    // Construct and register two Linear submodules.
    LeNet::fc1 = register_module("fc1", torch::nn::Linear(784, 64));
    LeNet::fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    LeNet::fc3 = register_module("fc3", torch::nn::Linear(32, 10));

    LeNet::relu = register_module("relu", torch::nn::ReLU());
  }

  torch::Tensor forward(torch::Tensor x) {
    x = x.view({x.size(0), 784});
    x = LeNet::fc1->forward(x);
    x = LeNet::relu(x);
    x = LeNet::fc2->forward(x);
    x = LeNet::relu(x);
    x = LeNet::fc3->forward(x);
    x = LeNet::relu(x);
    x = torch::log_softmax(x, /*dim=*/1);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  torch::nn::ReLU relu{nullptr};
};

#endif // MODEL_H