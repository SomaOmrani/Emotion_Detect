# # Use an official Python runtime as a parent image
# FROM python:3.11-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     portaudio19-dev \
#     git \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     alsa-utils \
#     ffmpeg \
#     libsm6 \
#     libxext6 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Install Python dependencies
# RUN pip install --upgrade pip
# COPY requirements.txt /app/requirements.txt
# RUN pip install -r requirements.txt

# # Ensure Git LFS is installed
# RUN apt-get update && apt-get install -y git-lfs
# RUN git lfs install --force

# # Pull the large files with Git LFS
# RUN git lfs pull

# # Expose the port the app runs on
# EXPOSE 8501

# # Run the app
# CMD ["streamlit", "run", "app.py"]


FROM python:3.11-slim

WORKDIR /app

COPY . /app

# Change the Debian mirror
RUN sed -i 's|http://deb.debian.org/debian|http://ftp.us.debian.org/debian|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    alsa-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y git-lfs
RUN git lfs install --force
RUN git lfs pull

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]

