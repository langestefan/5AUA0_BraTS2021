ARG UBUNTU_VERSION
FROM ubuntu:${UBUNTU_VERSION}

RUN apt update && apt install  openssh-server sudo -y

ARG PORT
ARG USER_ID
ARG USER_PW

# Add user, see: https://stackoverflow.com/a/49848507
RUN useradd -rm -d /home/${USER_ID} -s /bin/bash -g root -G sudo -u 1000 ${USER_ID} 

RUN echo ${USER_ID}:${USER_PW} | chpasswd

# not sure if needed: -g root -G sudo should be enough?
# RUN usermod -aG sudo ${USER_ID}
# RUN chown -R ${USER_ID}:root /home/${USER_ID}

RUN service ssh start

EXPOSE ${PORT}

CMD ["/usr/sbin/sshd","-D"]
