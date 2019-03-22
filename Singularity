BootStrap: docker
From: nvidia/cuda:9.0-devel-ubuntu16.04
# -----------------------------------------------------------------------------------
# This is a port of the Dockerfile maintained at https://github.com/uber/horovod


%environment
# -----------------------------------------------------------------------------------

    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs
    export LC_ALL=C
    export HOROVOD_GPU_ALLREDUCE=NCCL
    export HOROVOD_GPU_ALLGATHER=MPI
    export HOROVOD_GPU_BROADCAST=MPI
    export HOROVOD_NCCL_HOME=/usr/local/cuda/nccl
    export HOROVOD_NCCL_INCLUDE=/usr/local/cuda/nccl/include
    export HOROVOD_NCCL_LIB=/usr/local/cuda/nccl/lib 
    export PYTHON_VERSION=2.7
    export TENSORFLOW_VERSION=1.11.0

    export CUDNN_VERSION=7.3.1.20-1+cuda9.0
    export NCCL_VERSION=2.3.5-2+cuda9.0
    export PATH=/usr/local/mpich/install/bin/:${PATH}
    export LD_LIBRARY_PATH=/usr/local/mpich/install/lib/:${LD_LIBRARY_PATH}

%post
# -----------------------------------------------------------------------------------
# this will install all necessary packages and prepare the container

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
# Python 2.7 or 3.5 is supported by Ubuntu Xenial out of the box


    export PYTHON_VERSION=2.7
    export TENSORFLOW_VERSION=1.11.0

    export CUDNN_VERSION=7.3.1.20-1+cuda9.0
    export NCCL_VERSION=2.3.5-2+cuda9.0

    echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

    apt-get -y update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        mlocate \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev

    ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install TensorFlow, Keras and PyTorch and other g3dnet dependencies
    pip install tensorflow-gpu==${TENSORFLOW_VERSION} keras h5py 

    pip --no-cache-dir --disable-pip-version-check install --upgrade setuptools
    pip --no-cache-dir --disable-pip-version-check install future
    pip --no-cache-dir --disable-pip-version-check install 'matplotlib<3.0' # for python2.7
    pip --no-cache-dir --disable-pip-version-check install 'ipython<6.0'    # for python2.7
    pip --no-cache-dir --disable-pip-version-check install 'ipykernel<5.0'  # for python2.7
    pip --no-cache-dir --disable-pip-version-check install numpy wheel zmq six pygments pyyaml cython gputil psutil humanize h5py tqdm scipy seaborn tables
    pip --no-cache-dir --disable-pip-version-check install  pandas scikit-image scikit-learn Pillow opencv-python
    pip --no-cache-dir --disable-pip-version-check install jupyter notebook
    pip --no-cache-dir --disable-pip-version-check install transforms3d
    pip --no-cache-dir --disable-pip-version-check install pyamg


# Install the IB verbs
    apt install -y --no-install-recommends libibverbs*
    apt install -y --no-install-recommends ibverbs-utils librdmacm* infiniband-diags libmlx4* libmlx5* libnuma*

    # install MPICH
    wget -q http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz
    tar xf mpich-3.2.1.tar.gz
    rm mpich-3.2.1.tar.gz
    cd mpich-3.2.1
    # disable the addition of the RPATH to compiled executables
    # this allows us to override the MPI libraries to use those
    # found via LD_LIBRARY_PATH
    ./configure --prefix=/usr/local/mpich/install --disable-wrapper-rpath --disable-fortran
    make -j 4 install
    # add to local environment to build pi.c
    export PATH=$PATH:/usr/local/mpich//install/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/mpich//install/lib
    env | sort
    cd ..
    rm -rf mpich-3.2.1

    # nccl2
    git clone https://github.com/NVIDIA/nccl.git
    cd nccl;
    make -j src.build
    make pkg.redhat.build
    rpm -i build/pkg/rpm/x86_64/libnccl*
    cd -



# Install Horovod, temporarily using CUDA stubs
    ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir horovod && \
    ldconfig

# Set default NCCL parameters
    echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf


