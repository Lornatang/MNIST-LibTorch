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
#include <torch/torch.h>

using namespace std;

int main(int argc, const char *argv[]) {
  if (argc < 2) {
    cerr << "Usage: ./detect <path-to-image>\n";
    cerr << "Example: ./detect ../data/image.png ../assets/model_best.pth\n";
    return 0;
  }

  // choice GPU or CPU
  torch::Device device = select_device();

  auto model = std::make_shared<Net>();
  if (argv[2])
    torch::load(model, argv[2]);
  else
    torch::load(model, "../assets/model_best.pth");
  // check model is loaded.
  AT_ASSERT(model != nullptr);

  // move to GPU
  model->to(device);

  // load image
  cv::Mat image = cv::imread(argv[1]);
  if (image.empty()) {
    std::cerr << "Cant't load image.\n";
    return -1;
  }
  cv::Mat gray_image;
  cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
  cv::resize(gray_image, gray_image, cv::Size(28, 28));

  cout << "classes: " << int(classifier(gray_image, model, device))
       << endl;
  return 0;
}