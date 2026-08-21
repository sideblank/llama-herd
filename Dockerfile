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

# --- llama.cpp -------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS llama
ARG LLAMA_CPP_REF

RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} \
      https://github.com/ggml-org/llama.cpp.git /src/llama.cpp

# 86 = Ampere (3090), 89 = Ada (4090), 120 = Blackwell (5090).
# Only the library is built: the tools, examples, server and unified app are not used
# here, and the app target in particular pulls in the common library we do not need.
RUN cmake -S /src/llama.cpp -B /src/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="86;89;120" \
      -DLLAMA_BUILD_COMMON=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_TOOLS=OFF \
      -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_BUILD_UI=OFF \
      -DLLAMA_BUILD_APP=OFF \
      -DCMAKE_INSTALL_PREFIX=/opt/llama \
 && cmake --build /src/build --config Release -j "$(nproc)" \
 && cmake --install /src/build

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

# The runtime libraries sit beside the binary in the final image, so the loader is told to
# look there rather than in a system path.
ENV CGO_CFLAGS="-I/opt/llama/include"
ENV CGO_LDFLAGS="-L/opt/llama/lib -Wl,-rpath,\$ORIGIN/../lib"

RUN go build -trimpath \
      -ldflags "-s -w -X main.version=${VERSION} -X main.commit=${COMMIT} -X main.llamaCppRef=${LLAMA_CPP_REF}" \
      -o /out/bin/llama-herd ./cmd/llama-herd

# --- runtime ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=llama /opt/llama/lib /opt/llama-herd/lib
COPY --from=build /out/bin/llama-herd /opt/llama-herd/bin/llama-herd

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

ENTRYPOINT ["llama-herd"]
CMD ["serve", "--manifest", "/etc/llama-herd/manifest.json"]
