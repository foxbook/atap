FROM python:3.8.11

# Configure environment
ENV SHELL=/bin/bash

# Create non-root user
ARG USERNAME=python
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
	
# Force color prompt
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /home/$USERNAME/.bashrc

# Install useful OS packages
RUN apt-get update \
    && apt-get install -yq --no-install-recommends \
        htop \
        vim \
        tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install useful Python packages
RUN pip install --no-cache-dir pipenv

# Set project directory
ARG PROJECT_DIR=/home/$USERNAME/project

WORKDIR $PROJECT_DIR
COPY Pipfile $PROJECT_DIR
RUN chown -R $USER_UID:$USER_GID $PROJECT_DIR

# Set user
USER $USERNAME

# Install project dependencies
RUN pipenv install --dev && pipenv --clear
