# Pinned CUDA devel image for building and running llama-resident (design §3).
# Organizer builds this once; ranked jobs never pull or rebuild.
FROM nvidia/cuda@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
     cmake ninja-build \
  && rm -rf /var/lib/apt/lists/*
# Recorded on build host (ubuntu24.04 arm64, 2026-10-08):
#   cmake 3.28.3-1build1
#   ninja-build 1.11.1-2
