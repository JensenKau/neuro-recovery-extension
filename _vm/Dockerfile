FROM node:21 as base_node

FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

RUN apt-get update -y && apt-get install -y \
    curl make git \
    python3 python3-venv python3-pip

ENV NODE_VERSION=21.7.1
ENV YARN_VERSION=1.22.19
COPY --from=base_node /usr/local/bin /usr/local/bin
COPY --from=base_node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm

CMD ["/bin/bash"]