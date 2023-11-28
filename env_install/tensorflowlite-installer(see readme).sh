#!/bin/bash
# Creator of this tensorflowlite script: Liam Petrie (scotabroad) <liamdpetrie@vivaldi.net>
# This script is an adaptation of the tensorflow package from the AUR for Ubuntu
# Original AUR maintainer info down below
# Maintainer: Sven-Hendrik Haase <svenstaro@archlinux.org>
# Maintainer: Konstantin Gizdov (kgizdov) <arch@kge.pw>
# Contributor: Adria Arrufat (archdria) <adria.arrufat+AUR@protonmail.ch>
# Contributor: Thibault Lorrain (fredszaq) <fredszaq@gmail.com>

pkgbase=tensorflow-lite
pkgname=tensorflow-lite
pkgver=2.12.0
_pkgver=2.12.0
pkgrel=1
pkgdesc="Library for computation using data flow graphs for scalable machine learning"
url="https://www.tensorflow.org/"
license='APACHE'
arch='x86_64'
basedir=$(pwd)
srcdir="${basedir}"/tensorflow-${_pkgver}
buildir="${srcdir}"/tflite_build
pkgdir="${basedir}"/deb
#depends=('c-ares' 'pybind11' 'openssl' 'libpng' 'curl' 'giflib' 'icu' 'libjpeg-turbo' 'openmp')
sudo apt install -y libc-ares-dev pybind11-dev openssl libssl-dev libpng++-dev libpng-dev curl \
                    libgif-dev libicu-dev icu-devtools libjpeg-turbo8-dev lib64gomp1 libblis64-openmp-dev libomp-12-dev \
                    libxnnpack-dev libpthreadpool-dev libpsimd-dev libgemmlowp-dev libfxdiv-dev libfp16-dev libflatbuffers-dev \
                    flatbuffers-compiler libfarmhash-dev libfftw3-dev libeigen3-dev cpuinfo libcpuinfo-dev libabsl-dev
sudo apt install -y python3-numpy git python3-wheel python3-installer python3-setuptools python3-h5py python3-keras-applications \
                    python3-keras-preprocessing cython3 patchelf python3-requests gcc-12 python-is-python3 cmake

#NVIDA stuff, where XXX is GPU version
#sudo apt install libcub-dev libnvidia-compute-XXX nvidia-utils-XXX nvidia-cudnn
#(Ubuntu does not package nccl, and I do not have an NVIDIA account to download it from)

optdepends=('tensorboard: Tensorflow visualization toolkit')

# consolidate common dependencies to prevent mishaps
sudo apt install -y python3-termcolor python3-astor python3-gast python3-numpy python3-protobuf python3-absl \
                    python3-h5py python3-keras-applications python3-keras-preprocessing python3-astunparse \
                    python3-flatbuffers python3-typing-extensions python3-pip python3-pybind11
sudo pip install keras tensorflow-estimator opt_einsum google-pasta


get_pyver () {
  python -c 'import sys; print(str(sys.version_info[0]) + "." + str(sys.version_info[1]))'
}

check_dir() {
  # first make sure we do not break parsepkgbuild
  if ! command -v cp &> /dev/null; then
    >&2 echo "'cp' command not found. PKGBUILD is probably being checked by parsepkgbuild."
    if ! command -v install &> /dev/null; then
      >&2 echo "'install' command also not found. PKGBUILD must be getting checked by parsepkgbuild."
      >&2 echo "Cannot check if directory '${1}' exists. Ignoring."
      >&2 echo "If you are not running nacmap or parsepkgbuild, please make sure the PATH is correct and try again."
      >&2 echo "PATH should not be '/dummy': PATH=$PATH"
      return 0
    fi
  fi
  # if we are running normally, check the given path
  if [ -d "${1}" ]; then
    return 0
  else
    >&2 echo Directory "${1}" does not exist or is a file! Exiting...
    exit 1
  fi
}

prepare() {
  #Fetch Tensorflow
  cd ${basedir}
  wget -c https://github.com/tensorflow/tensorflow/archive/refs/tags/v${_pkgver}.tar.gz
  tar xvf v${_pkgver}.tar.gz
  #Fetch patches
  wget -c https://gitlab.archlinux.org/archlinux/packaging/packages/tensorflow/-/raw/main/fix-c++17-compat.patch
  wget -c  https://gitlab.archlinux.org/archlinux/packaging/packages/tensorflow/-/raw/main/tensorflow-2.10-sparse-transpose-op2.patch

  # Allow any bazel version
  echo "*" > tensorflow-${_pkgver}/.bazelversion

  # Get rid of hardcoded versions. Not like we ever cared about what upstream
  # thinks about which versions should be used anyway. ;) (FS#68772)
  sed -i -E "s/'([0-9a-z_-]+) .= [0-9].+[0-9]'/'\1'/" tensorflow-${_pkgver}/tensorflow/tools/pip_package/setup.py

  patch -Np1 -i "${srcdir}/tensorflow-2.10-sparse-transpose-op2.patch" -d tensorflow-${_pkgver}

  # These environment variables influence the behavior of the configure call below.
  export PYTHON_BIN_PATH=/usr/bin/python
  export USE_DEFAULT_PYTHON_LIB_PATH=1
  export TF_NEED_JEMALLOC=1
  export TF_NEED_KAFKA=1
  export TF_NEED_OPENCL_SYCL=0
  export TF_NEED_AWS=1
  export TF_NEED_GCP=1
  export TF_NEED_HDFS=1
  export TF_NEED_S3=1
  export TF_ENABLE_XLA=1
  export TF_NEED_GDR=0
  export TF_NEED_VERBS=0
  export TF_NEED_OPENCL=0
  export TF_NEED_MPI=0
  export TF_NEED_TENSORRT=0
  export TF_NEED_NGRAPH=0
  export TF_NEED_IGNITE=0
  export TF_NEED_ROCM=0
  # See https://github.com/tensorflow/tensorflow/blob/master/third_party/systemlibs/syslibs_configure.bzl
  export TF_SYSTEM_LIBS="boringssl,curl,cython,gif,icu,libjpeg_turbo,nasm,png,pybind11,zlib"
  export TF_SET_ANDROID_WORKSPACE=0
  export TF_DOWNLOAD_CLANG=0
  export TF_IGNORE_MAX_BAZEL_VERSION=1
  # Does tensorflow really need the compiler overridden in 5 places? Yes.
  export CC=gcc
  export CXX=g++
  export HOST_C_COMPILER=/usr/bin/${CC}
  export HOST_CXX_COMPILER=/usr/bin/${CXX}
  #Please enable to following lines if building for NVIDIA
  #export GCC_HOST_COMPILER_PATH=/opt/cuda/bin/gcc
  #export TF_NCCL_VERSION=$(pkg-config nccl --modversion | grep -Po '\d+\.\d+')
  #export NCCL_INSTALL_PATH=/usr
  #export TF_CUDA_CLANG=0  # Clang currently disabled because it's not compatible at the moment.
  #export CLANG_CUDA_COMPILER_PATH=/usr/bin/clang
  #export TF_CUDA_PATHS=/opt/cuda,/usr/lib,/usr
  #export TF_CUDA_VERSION=$(/opt/cuda/bin/nvcc --version | sed -n 's/^.*release \(.*\),.*/\1/p')
  #export TF_CUDNN_VERSION=$(sed -n 's/^#define CUDNN_MAJOR\s*\(.*\).*/\1/p' /usr/include/cudnn_version.h)
  # https://github.com/tensorflow/tensorflow/blob/1ba2eb7b313c0c5001ee1683a3ec4fbae01105fd/third_party/gpus/cuda_configure.bzl#L411-L446
  # according to the above, we should be specifying CUDA compute capabilities as 'sm_XX' or 'compute_XX' from now on
  # add latest PTX for future compatibility
  # Valid values can be discovered from nvcc --help
  #export TF_CUDA_COMPUTE_CAPABILITIES=sm_52,sm_53,sm_60,sm_61,sm_62,sm_70,sm_72,sm_75,sm_80,sm_86,sm_87,sm_89,sm_90,compute_90

  export BAZEL_ARGS="--config=mkl -c opt"
}

build() {
  echo "Building without cuda and without non-x86-64 optimizations"
  mkdir -p "${buildir}"
  cd "${buildir}"
  export CC_OPT_FLAGS="-march=x86-64"
  export TF_NEED_CUDA=0

  # https://github.com/tensorflow/tensorflow/issues/60577
  export CC=gcc
  export CXX=g++
  export CXXFLAGS="-fPIC"

  #./configure No need as not using bazel
  cmake \
    -DTFLITE_BUILD_EVALTOOLS=on \
    -DTFLITE_BUILD_SHARED_LIB=on \
    -DTFLITE_ENABLE_NNAPI=on \
    -DTFLITE_ENABLE_NNAPI_VERBOSE_VALIDATION=on \
    -DTFLITE_ENABLE_RUY=on \
    -DTFLITE_ENABLE_XNNPACK=on \
    -DTFLITE_PYTHON_WRAPPER_BUILD_CMAKE2=on \
    -DTFLITE_ENABLE_EXTERNAL_DELEGATE=on \
    ../tensorflow/lite
  cmake --build . -j$(nproc)
  PYTHON=python3 BUILD_NUM_JOBS=$(nproc) ../tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh native
 }

_package() {
  mkdir -p "${pkgdir}"
  # install libraries
  install -d "${pkgdir}"/usr/lib
  for lib in "${buildir}"/lib*.* #have some .so and .a libraries... why?
  do
    cp --no-preserve=ownership -d $lib "${pkgdir}"/usr/lib
  done

  # install headers
  install -d "${pkgdir}"/usr/include/tensorflow/lite
  cd ${srcdir}/tensorflow/lite
  cp --parents \
    $(find . -name "*.h*") \
    "${pkgdir}"/usr/include/tensorflow/lite
  install -d "${pkgdir}"/usr/include/tensorflow/core/public
  cd "${srcdir}"
  cp tensorflow/core/public/version.h ${pkgdir}/usr/include/tensorflow/core/public

  # install python-version to get all extra headers
  WHEEL_PACKAGE=$(find "${srcdir}" -name "tflite_runtime-*.whl")
  pip3 install --disable-pip-version-check -v \
    -t "${pkgdir}/usr/lib/python$(get_pyver)"/site-packages/tflite_runtime --no-cache-dir --no-deps \
    $WHEEL_PACKAGE

  # clean up unneeded files
  rm -rf "${pkgdir}"/usr/bin
  rm -rf "${pkgdir}"/usr/share

  # make sure no lib objects are outside valid paths
  local _so_srch_path="${pkgdir}/usr/include"
  check_dir "${_so_srch_path}"  # we need to quit on broken search paths
  find "${_so_srch_path}" -type f,l \( -iname "*.so" -or -iname "*.so.*" \) -print0 | while read -rd $'\0' _so_file; do
    # check if file is a dynamic executable
    ldd "${_so_file}" &>/dev/null && rm -rf "${_so_file}"
  done

  # install the rest of tensorflow
  chmod a+x tensorflow/lite/generate-pc.sh
  tensorflow/lite/generate-pc.sh --prefix=/usr --libdir=/usr/lib --version=${pkgver}
  install -Dm644 tensorflowlite.pc "${pkgdir}"/usr/lib/pkgconfig/tensorflowlite.pc
  install -Dm644 LICENSE "${pkgdir}"/usr/share/licenses/${pkgname}/LICENSE
}

package_tensorflow() {
  cd "${buildir}"
  _package
}

install_tensorflow() {
  sudo cp -rv "{pkgdir}"/* /
}

# Call functions
prepare
build
package_tensorflow
install_tensorflow
# vim:set ts=2 sw=2 et:
