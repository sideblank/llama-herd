# syntax=docker/dockerfile:1
#
# CUDA image for llama-herd.
#
# One image serves RTX 3090, 4090 and 5090 by compiling for all three compute
# capabilities. Blackwell (sm_120) needs CUDA 12.8 or newer, which is why the base is
# pinned this high — an older toolkit builds fine and then fails at runtime on a 5090.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=22.04
ARG LLAMA_CPP_REF=b10545
ARG GO_VERSION=1.24.0

# Decode-speed build options. Upstream defaults are conservative; these are the levers that
# matter for a multi-stream decode loop, and each is switchable so they can be A/B measured
# rather than assumed.
#
# CUDA_GRAPHS: upstream default OFF. A decode loop is thousands of small kernel launches per
#   second, and graphs replay a captured launch sequence instead of re-issuing each one. The
#   benefit is largest exactly where this engine operates — many streams, little work per step.
# FA_ALL_QUANTS: upstream default OFF, which compiles flash-attention kernels only for
#   MATCHED pairs (f16/f16, q4_0/q4_0, q8_0/q8_0, bf16/bf16). Mixed K and V precision — K at
#   q8 and V at q4, which saves a quarter of the cache since V tolerates less precision than
#   K — has no kernel without this and silently falls off the fast path.
# FORCE_MMQ: quantized matmul without a separate dequantization pass. Genuinely a trade
#   rather than a win, so it stays at the upstream default and is left to measurement.
ARG GGML_CUDA_GRAPHS=ON
ARG GGML_CUDA_FA_ALL_QUANTS=ON
ARG GGML_CUDA_FORCE_MMQ=OFF

# --- llama.cpp -------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS llama
ARG LLAMA_CPP_REF
ARG GGML_CUDA_GRAPHS
ARG GGML_CUDA_FA_ALL_QUANTS
ARG GGML_CUDA_FORCE_MMQ

RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} \
      https://github.com/ggml-org/llama.cpp.git /src/llama.cpp

# LLAMA_BUILD_COMMON is ON because the speculative implementation lives there. Driving a
# multi-token-prediction head needs the target's hidden states, and that function is not in
# the installed public header — it is in a staging header upstream marks work-in-progress and
# asks callers not to include. Going through common uses upstream's maintained implementation
# instead, which also covers all three head architectures rather than one.
#
# LLAMA_BUILD_MTMD builds libmtmd, the multimodal library behind vision and audio input.
# Its standalone hook exists for exactly this case — packaging the library for a language
# binding — and it fires only when common and tools are off, which is this configuration.
# It links ggml and llama alone; linking llama-common is a hard error upstream.
#
# 86 = Ampere (3090), 89 = Ada (4090), 120 = Blackwell (5090).
# Only the library is built: the tools, examples, server and unified app are not used
# here, and the app target in particular pulls in the common library we do not need.
# GGML_BACKEND_DL makes each backend a plugin loaded at run time instead of a hard
# NEEDED entry. Without it libggml.so requires libggml-cuda.so, which requires the host
# driver's libcuda.so.1 — so the binary cannot even print its version on a machine
# without a GPU, and a CPU-only deployment is impossible.
#
# GGML_NATIVE=OFF stops the CPU backend being compiled for the build machine's exact
# instruction set. A container scheduled onto an arbitrary node would otherwise die with
# an illegal instruction on any CPU older than the builder's.
RUN cmake -S /src/llama.cpp -B /src/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DGGML_BACKEND_DL=ON \
      -DGGML_NATIVE=OFF \
      -DGGML_CUDA=ON \
      -DGGML_CUDA_GRAPHS=${GGML_CUDA_GRAPHS} \
      -DGGML_CUDA_FA_ALL_QUANTS=${GGML_CUDA_FA_ALL_QUANTS} \
      -DGGML_CUDA_FORCE_MMQ=${GGML_CUDA_FORCE_MMQ} \
      -DCMAKE_CUDA_ARCHITECTURES="86;89;120" \
      -DLLAMA_BUILD_COMMON=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_TOOLS=OFF \
      -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_BUILD_UI=OFF \
      -DLLAMA_BUILD_APP=OFF \
      -DLLAMA_BUILD_MTMD=ON \
      -DCMAKE_INSTALL_PREFIX=/opt/llama \
 && cmake --build /src/build --config Release -j "$(nproc)" \
 && cmake --install /src/build

# The speculative shim: a pointer-only C ABI over common's C++ interface, which passes
# std::vector and references that cgo cannot bind. Built here, against the same llama.cpp
# tree, so the ABI cannot drift between the shim and the library it wraps.
COPY shim/ /shim/
RUN set -eux; \
    # Locate the common library rather than assuming a path: its name and directory depend
    # on whether the build produced shared or static libraries, and a wrong guess fails at
    # link time with a message about a missing file rather than about the assumption.
    LC="$(find /src/build -name 'libllama-common.*' -type f | head -1)"; \
    test -n "$LC" || { echo "common library not found — is LLAMA_BUILD_COMMON on?"; exit 1; }; \
    echo "linking against $LC"; \
    case "$LC" in \
      *.a) WHOLE="-Wl,--whole-archive $LC -Wl,--no-whole-archive" ;; \
      *)   WHOLE="$LC" ;; \
    esac; \
    g++ -O2 -fPIC -shared -std=c++17 \
        -I/src/llama.cpp/include -I/src/llama.cpp/ggml/include -I/src/llama.cpp/common \
        -I/shim \
        /shim/lhspec.cpp \
        -o /opt/llama/lib/liblhspec.so \
        -L"$(dirname "$LC")" -L/opt/llama/lib \
        $WHOLE \
        -lllama -lggml -lggml-base \
        -Wl,-rpath,'$ORIGIN' -Wl,--allow-shlib-undefined; \
    cp /shim/lhspec.h /opt/llama/include/; \
    # A shared common must ship too, or the runtime image loads a library whose dependency
    # is absent — a failure that appears only when the server starts.
    case "$LC" in *.so*) cp -P "$(dirname "$LC")"/libllama-common.so* /opt/llama/lib/ ;; esac

# --- llama-herd ------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build
ARG GO_VERSION
ARG LLAMA_CPP_REF

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" \
      | tar -C /usr/local -xz
ENV PATH=/usr/local/go/bin:$PATH

COPY --from=llama /opt/llama /opt/llama

WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal

ARG VERSION=dev
ARG COMMIT=none
ARG BUILD_DATE=unknown

# The runtime libraries sit beside the binary in the final image, so the loader is told to
# look there rather than in a system path.
ENV CGO_CFLAGS="-I/opt/llama/include"
# --allow-shlib-undefined is required, not cosmetic. libggml-cuda.so references CUDA
# driver symbols (cuMemMap, cuGetErrorString, ...) that live in libcuda.so, which is
# supplied by the host driver at run time and is absent from any build image. Without
# this the link fails on a machine that has no GPU — that is, on every build machine.
ENV CGO_LDFLAGS="-L/opt/llama/lib -Wl,-rpath,\$ORIGIN/../lib -Wl,--allow-shlib-undefined"

RUN go build -trimpath \
      -ldflags "-s -w -X main.version=${VERSION} -X main.commit=${COMMIT} -X main.date=${BUILD_DATE} -X main.llamaCppRef=${LLAMA_CPP_REF}" \
      -o /out/bin/llama-herd ./cmd/llama-herd

# --- runtime ---------------------------------------------------------------------------
# The base image, not the runtime image. The runtime image ships about 2.8 GB of CUDA
# libraries; static inspection of the CUDA backend shows it names only the runtime, cuBLAS
# and NCCL, plus the host driver supplied at run time. cuFFT, cuSPARSE, cuSOLVER, cuRAND,
# NPP, nvJPEG and OpenCL are never loaded.
#
# This matters beyond disk. A scheduled worker cold-pulls the whole image before the
# container starts, so a smaller image is a shorter window in which a spot node can vanish
# mid-pull — which is a reliability property, not just a speed one.
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

# Exactly the CUDA libraries the backend names.
#
# These are copied with `cp -P` rather than COPY because COPY dereferences symlinks. A CUDA
# library ships as libfoo.so -> libfoo.so.12 -> libfoo.so.12.8.4.1, and COPY turns those three
# names into three independent copies of the same file. Measured: cuBLASLt alone went in at
# 700 MB and landed as 2.26 GB, and the whole image carried 2.3 GB of duplicates.
#
# That is not a disk complaint. A scheduled worker cold-pulls the entire image before the
# container starts, so every duplicated gigabyte widens the window in which a spot node can
# vanish mid-pull — and on a consumer uplink that window is already the main reason a
# deployment never reaches the point of downloading its model.
RUN --mount=from=llama,source=/usr/local/cuda/lib64,target=/cudalib,ro \
    --mount=from=llama,source=/usr/lib/x86_64-linux-gnu,target=/syslib,ro \
    set -eux; \
    mkdir -p /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu; \
    cp -P /cudalib/libcudart.so* /cudalib/libcublas.so* /cudalib/libcublasLt.so* \
          /usr/local/cuda/lib64/; \
    cp -P /syslib/libnccl.so* /usr/lib/x86_64-linux-gnu/; \
    # Prove the links survived: if these are regular files the duplication is back.
    test -L /usr/local/cuda/lib64/libcublasLt.so; \
    test -L /usr/lib/x86_64-linux-gnu/libnccl.so.2
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# libgomp1 is required, not optional: ggml's CPU backend links OpenMP, and no CUDA base
# ships it. Without it the binary dies at startup with
# "libgomp.so.1: cannot open shared object file" — a failure that only appears when the
# image is actually run, never during the build.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=llama /opt/llama/lib /opt/llama-herd/lib
# With GGML_BACKEND_DL the backends install to bin/, not lib/, and they are loaded by
# path at run time rather than linked. ggml_backend_load_all searches the directory of
# the running executable, so they go beside the binary. Copying only lib/ silently drops
# them, and the result looks healthy while finding no devices at all.
COPY --from=llama /opt/llama/bin /opt/llama-herd/bin
COPY --from=build /out/bin/llama-herd /opt/llama-herd/bin/llama-herd
COPY docker-entrypoint.sh /opt/llama-herd/bin/docker-entrypoint.sh

# ggml loads its CUDA backend as a plugin at runtime rather than linking it, so the
# library directory must be discoverable or the GPU silently does not appear.
ENV LD_LIBRARY_PATH=/opt/llama-herd/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/llama-herd/bin:$PATH

# Models are mounted or downloaded at run time; the image carries none.
VOLUME ["/models"]
WORKDIR /opt/llama-herd

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["serve", "--manifest", "/etc/llama-herd/manifest.json"]
