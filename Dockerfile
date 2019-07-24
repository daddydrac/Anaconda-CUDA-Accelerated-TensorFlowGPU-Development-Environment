FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# Core Linux Deps
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --fix-missing --no-install-recommends \
        build-essential \
    	cmake \
        curl \
	binutils \
	gdb \
        git \
	freeglut3 \
	freeglut3-dev \
	libxi-dev \
	libxmu-dev \
	gfortran \
        pkg-config \
	python-numpy \
	python-dev \
	python-setuptools \
	libboost-python-dev \
	libboost-thread-dev \
        pbzip2 \
        rsync \
        software-properties-common \
        libboost-all-dev \
        libopenblas-dev \ 
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
	libgraphicsmagick1-dev \
        libavformat-dev \
        libhdf5-dev \
        libpq-dev \
	libgraphicsmagick1-dev \
	libavcodec-dev \
	libgtk2.0-dev \
	liblapack-dev \
        liblapacke-dev \
	libswscale-dev \
	libcanberra-gtk-module \
        libboost-dev \
	libboost-all-dev \
        libeigen3-dev \
	wget \
        vim \
        qt5-default \
        unzip \
	zip \ 
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*  && \
    apt-get clean && rm -rf /tmp/* /var/tmp/*

# Install TensorRT (TPU Access)
RUN apt-get update && \
        apt-get install -y nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0 && \
        apt-get update && \
        apt-get install -y libnvinfer5=5.0.2-1+cuda10.0

RUN file="$(ls -1 /usr/local/)" && echo $file


# Install Anaconda
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
/bin/bash Miniconda3-latest-Linux-x86_64.sh -f -b -p /opt/conda && \
rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH /opt/conda/bin:$PATH


# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

ARG PYTHON=python3
ARG PIP=pip3

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools
    
RUN pip install pycuda

RUN conda update -n base -c defaults conda

# Install TF for GPU
RUN conda create -n tensorflow_gpuenv tensorflow-gpu
RUN /bin/bash -c conda activate tensorflow_gpuenv

RUN conda install -c anaconda jupyter 

RUN conda update conda
RUN conda install numba
RUN conda install -c anaconda cupy 
RUN conda install -c anaconda ipykernel 
RUN conda install -c conda-forge featuretools
RUN conda install -c anaconda scikit-learn 
RUN conda install -c anaconda future 
RUN conda install -c conda-forge dask 
RUN conda install -c conda-forge xgboost 
RUN conda install -c anaconda seaborn 
RUN conda install -c anaconda ipython 
RUN conda install -c anaconda keras-gpu

WORKDIR /

RUN mkdir -p /tf/tensorflow-tutorials && chmod -R a+rwx /tf/
RUN mkdir /.local && chmod a+rwx /.local
RUN chmod -R 777 /.local
RUN apt-get install -y --no-install-recommends wget
WORKDIR /tf/tensorflow-tutorials
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/basic_classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/basic_text_classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/overfit_and_underfit.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/save_and_restore_models.ipynb

RUN apt-get autoremove -y && apt-get remove -y wget
RUN conda install -c conda-forge tensorboard 
RUN conda install ipykernel jupyter
RUN python -m ipykernel install --user --name tf-gpu --display-name "TensorFlow-GPU"

WORKDIR /tf
EXPOSE 8888 6006

RUN useradd -ms /bin/bash container_user
RUN python3 -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.custom_display_url='http://localhost:8888'"]
