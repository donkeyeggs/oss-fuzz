FROM gcr.io/oss-fuzz-base/base-builder-python

RUN git clone --depth 1 --branch master https://github.com/BVLC/caffe.git

WORKDIR caffe

COPY build.sh $SRC/