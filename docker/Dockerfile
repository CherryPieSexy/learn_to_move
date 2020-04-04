FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Install prerequested
RUN apt-get -y update && apt-get -y install \
    vim htop git wget \
    libglu1 libsm6 libxi6 \
    sudo software-properties-common cmake gdb unzip

# Copy requirements
COPY ./requirements.txt /root/requirements.txt

# Install anaconda
RUN . ~/.bashrc && \
    wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh -O ~/Anaconda3.sh && \
    bash ~/Anaconda3.sh -b && \
    echo 'export PATH="/root/anaconda3/bin:$PATH"' >> ~/.bashrc && \
    rm ~/Anaconda3.sh
RUN . ~/.bashrc && \
    conda update --all && \
    conda create -n opensim-rl -c kidzik opensim python=3.6

RUN . ~/.bashrc && \
    . /root/anaconda3/etc/profile.d/conda.sh && \
    conda activate opensim-rl && \
    conda install -c conda-forge lapack git && \
    pip install --upgrade pip && \
    pip install -U -r /root/requirements.txt

# Set normal TZ (mostly for logs)
RUN ln -sf /usr/share/zoneinfo/Europe/Moscow /etc/localtime
