# Anaconda with TensorFlowGPU, Keras-GPU, CuPy and NVIDIA CUDA X #

Containerized, reproducible, development environment with Anaconda, NVIDIA CUDA 10.1, TensorFlow-GPU, Keras-GPU, Dask, CuPy (GPU Accelerated drop in Numpy replacement), and PyCUDA. 

-----------------------------------

#### Anaconda + Tensorflow: CUDA enabled GPU Machine Learning Development Environment ####
<p style="display:table;">
<img align="left" src="https://avatars2.githubusercontent.com/u/1158637?s=200&v=4" width="31%" height="auto" />
<img align="left" src="https://avatars2.githubusercontent.com/u/1728152?s=200&v=4" width="31%" height="auto" />
<img align="left" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/tensorflow/tensorflow.png" width="31%" height="auto" />
</p>

-----------------------------------

#### Features ####

* [Anaconda](https://www.anaconda.com/why-anaconda/): Anaconda is a distribution of Python for scientific computing

* [TensorFlow for GPU v1.13.1](https://www.tensorflow.org/install/gpu): GPU enabled Machine Learning framework 

* [TensorBoard](https://www.datacamp.com/community/tutorials/tensorboard-tutorial): Understand, debug, and optimize, located on ` localhost:6006  `

* [Keras-GPU](https://keras.io/): Keras: The Python Deep Learning library for GPUs

* [CuPy:latest](https://cupy.chainer.org/): GPU accelerated drop in replacement for numpy


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
   
   1. [CPU/GPU Benchmark](https://github.com/joehoeller/ /blob/master/apps/gpu_benchmarks/benchmark.py)
   
   2. [Tensorboard to understand & debug neural networks](https://github.com/joehoeller/ /blob/master/apps/gpu_benchmarks/tensorboard.py)


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




