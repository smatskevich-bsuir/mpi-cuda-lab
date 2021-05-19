FROM nvidia/cuda:11.3.0-base-ubuntu20.04

# author
MAINTAINER Stanislau Matskevich

ENV USER=stanislau
ENV HOME=/home/${USER} 

ENV DEBIAN_FRONTEND=noninteractive

# extra metadata
LABEL version="0.1"
LABEL description="Docker image with CUDA and MPI"

# update sources list
RUN apt-get clean
RUN apt-get update

# install app runtimes and modules
RUN apt install -qq -y --no-install-recommends openssh-server mpich

# cleanup
RUN apt-get -qy autoremove

RUN useradd -m ${USER}
#RUN usermod -aG sudo ${USER}

ADD ssh/mpi_rsa.pub ${HOME}/.ssh/authorized_keys

RUN chmod -R 600 ${HOME}/.ssh/*
RUN chown -R ${USER}:${USER} ${HOME}/.ssh

RUN mkdir -p /var/run/sshd

EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]