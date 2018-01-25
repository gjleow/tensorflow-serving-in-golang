# Code Examples

### Setting up your local dev env (Mac OSX)
```
brew update
brew install pyenv
make env
```

### Cleaning up
```
make remove_env
brew uninstall pyenv
```

### Installing TensorFlow for Go
TensorFlow for Go depends on the TensorFlow C library. Take the following steps to install this library and enable TensorFlow for Go:
```
TF_TYPE="cpu" # Change to "gpu" for GPU support
 TARGET_DIRECTORY='/usr/local'
 curl -L \
   "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-$(go env GOOS)-x86_64-1.4.1.tar.gz" |
 sudo tar -C $TARGET_DIRECTORY -xz
 ```
Go get:
```
go get github.com/tensorflow/tensorflow/tensorflow/go
```

[More information](https://www.tensorflow.org/install/install_go)

### Build model and save it
```
cd tensorflow/example
python linear_model.py
```

### Run pred/infer from model
```
go run run_linear.go
```
