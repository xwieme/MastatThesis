FROM ubuntu

# Install pip
RUN apt update
RUN apt install python3-pip -y

ENTRYPOINT bash
