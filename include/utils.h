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

#ifndef UTILS_H
#define UTILS_H

#include "model.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <torch/cuda.h>
#include <torch/optim.h>
#include <unistd.h>
#include <vector>

torch::Device select_device();

template <typename DataLoader>
double train(size_t epoch, std::shared_ptr<LeNet> model, torch::Device device,
             DataLoader &data_loader, torch::optim::Optimizer &optimizer,
             size_t dataset_size) {
  // set train mode
  model->train();
  size_t batch_index = 0;
  // Iterate data loader to yield batches from the dataset
  for (auto &batch : data_loader) {
    auto images = batch.data.to(device);
    auto targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model->forward(images);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));

    // Compute gradients
    loss.backward();
    // Update the parameters
    optimizer.step();

    if (++batch_index % 10 == 0) {
      std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f", epoch,
                  batch_index * batch.data.size(0), dataset_size,
                  loss.template item<float>());
    }
  }
}

template <typename DataLoader>
std::vector<double> evaluate(std::shared_ptr<LeNet> model, torch::Device device,
                             DataLoader &data_loader, size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model->eval();

  // define value.
  double loss = 0.;
  double accuracy;
  size_t correct = 0;
  std::vector<double> result;

  for (const auto &batch : data_loader) {
    auto data = batch.data.to(device);
    auto targets = batch.target.to(device);
    auto output = model->forward(data);
    loss += torch::nll_loss(output, targets, {}, torch::Reduction::Sum)
                .template item<double>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }
  loss /= dataset_size;
  accuracy = static_cast<double>(correct) / dataset_size;

  std::printf("\nTest set: Average loss: %.4f | Accuracy: %.3f\n", loss,
              accuracy);

  result.push_back(loss);
  result.push_back(accuracy);

  return result;
}

double classifier(cv::Mat &image, const std::shared_ptr<LeNet> &model,
                  torch::Device device);

#endif // UTILS_H