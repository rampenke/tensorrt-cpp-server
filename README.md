# tensorrt-cpp-server
[TensorRT README](https://github.com/NVIDIA/TensorRT/).

A tensorrt example server using crow

## Cuda versions
Cuda-12.4, TensorRT 10.0.1.
Tested on RTX 3090

## Build using cmake
```
mkdir build
cd build
cmake ..
make
```
## Build directly with g++

```
g++ src/mnist.cpp src/server.cpp -I/usr/lib/x86_64-linux-gnu -L /usr/lib/x86_64-linux-gnu `pkg-config --cflags --libs cuda-12.4` `pkg-config --cflags --libs cudart-12.4` -lnvinfer -lnvonnxparser -pthread
```

## Testing

### Mnist model and test files
Download tensorrt package from NVidia developer site and locate the data folder containing mnist onnx model and test pgmp files. Copy the data folder to the  folder containing server bimary.

https://developer.nvidia.com/tensorrt


Copy mnist.onx file to a folder named data located along with server binary

Curl command for REST API to send a pgmp file containing a digit and get the inference result:
```
curl -X POST localhost:18080/api/upload   -H "Content-Type: multipart/form-data"   -F "file=@5.pgm"
```