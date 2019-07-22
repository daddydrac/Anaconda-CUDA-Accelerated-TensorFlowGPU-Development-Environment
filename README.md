# Anaconda with TensorFlow-GPU and NVIDIA CUDA X #

Containerized, reproducible, development environment with Anaconda, NVIDIA CUDA 10.1, TensorFlow-GPU, Keras-GPU, Dask, CuPy (GPU Accelerated drop in Numpy replacement), and PyCUDA. 

-----------------------------------

#### Anaconda + Tensorflow: CUDA enabled GPU Machine Learning Development Environment ####
<p style="display:table;">
<img align="left" src="https://avatars2.githubusercontent.com/u/1158637?s=200&v=4" width="19%" height="auto" />
<img align="left" src="https://avatars2.githubusercontent.com/u/1728152?s=200&v=4" width="19%" height="auto" />
<img align="left" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/tensorflow/tensorflow.png" width="31%" height="auto" />
<img align="left" src="https://numba.pydata.org/_static/numba-blue-horizontal-rgb.svg" width="19%" height="auto" />
<img align="left" src="data:image/png;base64,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" width="19%" height="auto" />
<img align="left" src="https://camo.githubusercontent.com/cfcfc32dae79f7857d760a358227665a054b5583/68747470733a2f2f7777772e66656174757265746f6f6c732e636f6d2f77702d636f6e74656e742f75706c6f6164732f323031372f31322f466561747572654c6162732d4c6f676f2d54616e676572696e652d3830302e706e67" width="19%" height="auto" />
</p>

-----------------------------------

#### Features ####

* [Anaconda](https://www.anaconda.com/why-anaconda/): Anaconda is a distribution of Python for scientific computing

* [TensorFlow for GPU v1.13.1](https://www.tensorflow.org/install/gpu): GPU enabled Machine Learning framework 

* [TensorBoard](https://www.datacamp.com/community/tutorials/tensorboard-tutorial): Understand, debug, and optimize, located on ` localhost:6006  `, [Official Docs](https://www.tensorflow.org/guide/summaries_and_tensorboard)

* [Keras-GPU](https://keras.io/): Keras: The Python Deep Learning library for GPUs

* [CuPy:latest](https://cupy.chainer.org/): GPU accelerated drop in replacement for numpy

* [Numba](https://numba.pydata.org/): Numba also works great with Jupyter notebooks for interactive computing, and with distributed execution frameworks, like Dask and Spark, allows you pipe functions to be executed on GPU, etc


### Distributed Feature Engineering 

* [Dask Distributed](https://dask.org/): Distributed ingestion of data/See [Scaling Python with Dask](https://www.slideshare.net/secret/fi010O0yhOEqZi)

* [Feature Tools](https://docs.featuretools.com/): Automted feature engineering


### CUDA for GPU/TPU Enablement

* [NVIDIA TensorRT inference accelerator and CUDA 10](https://developer.nvidia.com/tensorrt): CUDA + TPUs makes you awesome

* [PyCUDA 2019](https://mathema.tician.de/software/pycuda/): Comes with Anaconda Accelerate ([See here](https://www.scivision.dev/install-cuda-accelerate-for-anaconda-python/)); Python interface for direct access to GPU or TPU

* [cuDNN7.4.1.5 for deeep learning in CNN's](https://developer.nvidia.com/cudnn): GPU-accelerated library of primitives for deep neural networks

-----------------------------------------------------------

### Good to know
* Hot Reloading of Docker Container: code updates will automatically update in container from /apps folder.
* TensorBoard is on localhost:6006 and GPU enabled Jupyter is on localhost:8888.
* Python 3.7
* Only Tesla Pascal and Turing GPU Architecture are supported 
* Test with synthetic data that compares GPU to CPU benchmark, and Tensorboard example:
   
   1. [CPU/GPU Benchmark](https://github.com/joehoeller/Anaconda-CUDA-Accelerated-TensorFlowGPU-Development-Environment/tree/master/apps/gpu_benchmarks/benchmark.py)
   
   2. [Tensorboard to understand & debug neural networks](https://github.com/joehoeller/Anaconda-CUDA-Accelerated-TensorFlowGPU-Development-Environment/tree/master/apps/gpu_benchmarks/tensorboard.py)


-------------------------------------------------------------

### Before you begin (This might be optional) ###

Link to nvidia-docker2 install: [Tutorial](https://medium.com/@sh.tsang/docker-tutorial-5-nvidia-docker-2-0-installation-in-ubuntu-18-04-cb80f17cac65)

You must install nvidia-docker2 and all it's deps first, assuming that is done, run:


 ` sudo apt-get install nvidia-docker2 `
 
 ` sudo pkill -SIGHUP dockerd `
 
 ` sudo systemctl daemon-reload `
 
 ` sudo systemctl restart docker `
 

How to run this container:


### Step 1 ###

` docker build -t <container name> . `  < note the . after <container name>


### Step 2 ###

Run the image, mount the volumes for Jupyter and app folder for your fav IDE, and finally the expose ports `8888` for TF1, and `6006` for TensorBoard.


` docker run --rm -it --runtime=nvidia --user $(id -u):$(id -g) --group-add container_user --group-add sudo -v "${PWD}:/apps" -v $(pwd):/tf/notebooks  -p 8888:8888 -p 0.0.0.0:6006:6006  <container name> `


### Step 3: Check to make sure GPU drivers and CUDA is running ###

- Exec into the container and check if your GPU is registering in the container and CUDA is working:

- Get the container id:

` docker ps `

- Exec into container:

` docker exec -u root -t -i <container id> /bin/bash `

- Check if NVIDIA GPU DRIVERS have container access:

` nvidia-smi `

- Check if CUDA is working:

` nvcc -V `


### Step 4: How to launch TensorBoard ###

(It helps to use multiple tabs in cmd line, as you have to leave at least 1 tab open for TensorBoard@:6006)

- Demonstrates the functionality of TensorBoard dashboard


- Exec into container if you haven't, as shown above:


- Get the `<container id>`:
 

` docker ps `


` docker exec -u root -t -i <container id> /bin/bash `


- Then run in cmd line:


` tensorboard --logdir=//tmp/tensorflow/mnist/logs `


- Type in: ` cd / ` to get root.

Then cd into the folder that hot reloads code from your local folder/fav IDE at: `/apps/apps/gpu_benchmarks` and run:


` python tensorboard.py `


- Go to the browser and navigate to: ` localhost:6006 `



### Step 5: Run tests to prove container based GPU perf ###

- Demonstrate GPU vs CPU performance:

- Exec into the container if you haven't, and cd over to /tf/notebooks/apps/gpu_benchmarks and run:

- CPU Perf:

` python benchmark.py cpu 10000 `

- CPU perf should return something like this:

`Shape: (10000, 10000) Device: /cpu:0
Time taken: 0:00:03.934996`

- GPU perf:

` python benchmark.py gpu 10000 `

- GPU perf should return something like this:

`Shape: (10000, 10000) Device: /gpu:0
Time taken: 0:00:01.032577`


--------------------------------------------------


### Known conflicts with nvidia-docker and Ubuntu ###

AppArmor on Ubuntu has sec issues, so remove docker from it on your local box, (it does not hurt security on your computer):

` sudo aa-remove-unknown `

--------------------------------------------------

If building impactful data science tools is important to you or your business, please get in touch.

#### Contact
Email: joehoeller@gmail.com




