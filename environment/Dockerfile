# hash:sha256:c4871197360886537b2a051550c767c74ee6e4b277e83aeaa247e79a681159de
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=soumith --channel=fastai --channel=psi4 --channel=main --channel=pytorch --channel=anaconda --channel=conda-forge --channel=defaults \
        basemap==1.2.2 \
        captum==0.5.0 \
        dask==2021.10.0 \
        jupyter==1.0.0 \
        jupyterlab==3.3.2 \
        matplotlib==3.2.1 \
        netcdf4==1.5.3 \
        numpy==1.21.2 \
        pandas==1.3.5 \
        pillow==6.1.0 \
        pysteps==1.3.2 \
        python==3.7.4 \
        torchvision==0.2.1 \
        xarray==0.20.1 \
        xclim==0.33.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    pytorch-lightning==1.5.10

COPY postInstall /
RUN /postInstall
