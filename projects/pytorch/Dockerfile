FROM gcr.io/oss-fuzz-base/base-builder-python

#RUN git clone --depth 1 --branch master https://github.com/pytorch/pytorch.git

WORKDIR pytorch

RUN python3 -m pip install torch 

RUN python3 -m pip install atheris hypothesis numpy 

#RUN python3 -m pip install sqlalachemy


COPY build.sh fuzzer $SRC/
#COPY torch-1.11.0-cp38-cp38-manylinux1_x86_64.whl /src/pytorch