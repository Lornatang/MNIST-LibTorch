# MNIST-LibTorch

An example of MNIST handwritten number recognition for LibTorch written by pure CPP

### Quick start

#### download code
```bash
$ git clone https://github.com/Lornatang/MNIST-LibTorch.git
$ cd MNIST-LibTorch
$ mkdir build
$ cd build
$ cmake ..
$ make
```

#### Download dataset
```bash
$ cd <repo>/data
$ bash download.sh
```

#### Train
```bash
$ cd <repo>/build
$ ./train ../data
```

#### Inference
```bash
$ cd <repo>/build
$ ./eval ../data
```

#### Detect image
```bash
$ cd <repo>/build
$ ./detect ../data/image.png ../checkpoint/model_best.pth
```

### Thanks
Thank my parents for their great support for my work!
