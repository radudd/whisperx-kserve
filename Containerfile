# Use Red Hat UBI 9 with Python 3.11
FROM registry.access.redhat.com/ubi9/python-311:latest

# Switch to root for system package installation
USER 0

# Download and install ffmpeg from BtbN builds (shared version)
RUN curl -L https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl-shared.tar.xz \
    -o /tmp/ffmpeg.tar.xz \
    && tar -xf /tmp/ffmpeg.tar.xz -C /tmp \
    && cp /tmp/ffmpeg-master-latest-linux64-gpl-shared/bin/ffmpeg /usr/local/bin/ \
    && cp /tmp/ffmpeg-master-latest-linux64-gpl-shared/bin/ffprobe /usr/local/bin/ \
    && mkdir -p /usr/local/lib64/ffmpeg \
    && cp -r /tmp/ffmpeg-master-latest-linux64-gpl-shared/lib/* /usr/local/lib64/ffmpeg/ \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg*


# Add ffmpeg libs to library path and run ldconfig
RUN echo "/usr/local/lib64/ffmpeg" > /etc/ld.so.conf.d/ffmpeg.conf && \
    ldconfig

# Verify ffmpeg works
RUN ffmpeg -version && ffprobe -version


# Set working directory
WORKDIR /opt/app-root/src

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install WhisperX and KServe dependencies
RUN pip install --no-cache-dir \
    git+https://github.com/m-bain/whisperx.git \
    kserve \
    torch \
    torchaudio

# Copy the predictor code
COPY whisperx_predictor.py /opt/app-root/src/whisperx_predictor.py

# Create directory for model cache
RUN mkdir -p /opt/app-root/src/.cache && \
    chown -R 1001:0 /opt/app-root/src/.cache && \
    chmod -R g=u /opt/app-root/src/.cache

# Set environment variables
ENV HOME=/opt/app-root/src
ENV HF_HOME=/opt/app-root/src/.cache/huggingface
ENV TORCH_HOME=/opt/app-root/src/.cache/torch

# Switch back to non-root user (OpenShift requirement)
USER 1001

# Expose KServe default port
EXPOSE 8080

# Run the KServe model server
ENTRYPOINT ["python", "-m", "whisperx_predictor"]
