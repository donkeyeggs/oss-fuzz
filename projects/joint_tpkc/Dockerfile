FROM gcr.io/oss-fuzz-base/base-builder-python
#RUN git clone --depth 1 --branch master https://github.com/pytorch/pytorch.git

WORKDIR joint_tpkc

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install atheris
RUN python3 -m pip install hypothesis
RUN python3 -m pip install numpy
RUN python3 -m pip install torch
RUN python3 -m pip install tensorflow
RUN python3 -m pip install paddlepaddle==2.3.0 -i https://mirror.baidu.com/pypi/simple
#RUN python3 -m pip install caffe2
#RUN git clone https://github.com/Qengineering/caffe.git 
#RUN apt-get install --assume-yes libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev \
#&& apt-get install --assume-yes libhdf5-serial-dev protobuf-compiler \
#&& apt-get install --assume-yes --no-install-recommends libboost-all-dev \
#&& apt-get install --assume-yes libopenblas-dev liblapack-dev libatlas-base-dev \
#&& apt-get install --assume-yes libgflags-dev libgoogle-glog-dev liblmdb-dev 
#RUN find / -name caffe \
#&& cd /src/joint_tpkc/caffe/python \
#&& python3 -m pip install --upgrade python-dateutil\
#&& for req in $(cat requirements.txt); do pip3 install $req ; done 
#RUN cd /src/joint_tpkc/caffe \
#&& make clean \
#&& make test -j8 \
#&& make runtest -j8 \
#&& make pycaffe -j8 \
#&& export PYTHONPATH=/root/caffe/python:$PYTHONPATH \
#&& source ~/.bashrc 

COPY build.sh fuzzer $SRC/