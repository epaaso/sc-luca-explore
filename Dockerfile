FROM python:3.11

SHELL ["/bin/bash", "-c"]

# For editing in shell and pre-reqs for scArches and R
RUN apt update && apt install -y --no-install-recommends nano npm nodejs gfortran
 
# For cell annotation runos for almos all models in the notebooks
RUN pip install scarches==0.6.1

# For autotuning reference atlas, we need the main, that is not yet implemented in latest ver  1.15.0. It is implemented in 1.2.0
RUN pip install git+https://github.com/scverse/scvi-tools@f8811ad999d470e9d589520496905ae0328b1402 && pip install ray[tune]==2.23.0

# Jax Cuda for faster Nueral nets
RUN pip install jax[cuda] torch==2.4.0 flax==0.8.5 chex==0.1.86

# Assure pandas <2 because fo the saved models we have
RUN pip install pandas==1.5.3

# For notebooks
RUN pip install jupyterlab jupyterlab-git ipywidgets
RUN echo "alias jl='jupyter-lab --no-browser --ip=0.0.0.0 --allow-root /root/host_home'" >> ~/.bashrc

# For visualizing and builidng networks, with holovies also
RUN apt update && apt install -y --no-install-recommends libgraphviz-dev graphviz
RUN pip install pygraphviz==1.11 networkx==3.1 git+ssh://git@github.com/epaaso/holoviews.git@sankey-node datashader==0.16.3 scikit-image

# R from source is better for portability
ARG RVER=4.4.1
RUN wget https://cran.rstudio.com/src/base/R-4/R-$RVER.tar.gz
RUN tar xvfz R-$RVER.tar.gz && rm R-$RVER.tar.gz
WORKDIR R-$RVER
RUN ./configure --enable-R-shlib --with-cairo --with-libpng --prefix=/opt/R/
RUN make && make install
WORKDIR /opt/R
RUN rm -rf /opt/R/R-$RVER
ENV PATH="/opt/R/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/R/lib/R/lib:${LD_LIBRARY_PATH}"
RUN echo 'PATH="/opt/R/bin:${PATH}"' >> ~/.bashrc
RUN echo 'LD_LIBRARY_PATH="/opt/R/lib/R/lib:${LD_LIBRARY_PATH}"' >> ~/.bashrc

RUN echo "/opt/R/lib/R/lib" >> /etc/ld.so.conf.d/RLibs.conf && ldconfig

# R libraries
RUN apt install -y --no-install-recommends libharfbuzz-dev libfribidi-dev libhdf5-dev
RUN cat ~/.bashrc

RUN Rscript -e "update.packages(ask=FALSE, repos='https://cran.itam.mx/')"
RUN Rscript -e "install.packages(c('devtools', 'gam', 'RColorBrewer', 'BiocManager', 'IRkernel','png', 'hdf5r', 'Seurat', 'Cairo'), repos='https://cran.itam.mx/')"
RUN Rscript -e "IRkernel::installspec(user = FALSE)"
RUN Rscript -e "BiocManager::install(c('sparseMatrixStats', 'SparseArray', 'DelayedMatrixStats','scuttle', 'scry', 'edgeR', 'DropletUtils',  'biomaRt', 'glmGamPoi'))"
# RUN Rscript -e "devtools::install_github('MatteoBlla/PsiNorm')"

# For running R in python kernel
RUN pip install rpy2==3.5.14 anndata2ri==1.3.1

# Ikarus with sparse matrix handling
# RUN pip install git+https://github.com/epaaso/ikarus.git

# For opening xls files
RUN pip install openpyxl

# InferCNV for tumor annotation
# RUN pip install Cython==0.29.33 tables==3.9.1 infercnvpy==0.4.3
# WORKDIR ~
# RUN git clone https://github.com/cvanelteren/forceatlas2
# RUN cd forceatlas2 && echo "from forceatlas2 import *" >> fa2/__init__.py && pip install .

# For scFusion super lengthy tumor annotation
# WORKDIR ~
# RUN wget https://github.com/alexdobin/STAR/raw/2.7.10b/bin/Linux_x86_64_static/STAR
# RUN mv STAR /usr/bin && chmod +x STAR
# RUN apt install -y --no-install-recommends samtools bedtools
# RUN pip install pyensembl==2.2.9 tensorflow=2.15.0 keras=2.15.0
# RUN Rscript -e "install.packages(c('stringr'), repos='https://cran.itam.mx/')"

# JAVA for networks
RUN apt install -y --no-install-recommends default-jre

# Papermill to generalize notebooks
RUN pip install papermill==2.5.0

# Gprofiler for GO enrichment and GSEApy for enrichment agaisnt other databses
RUN pip install gprofiler-official==1.0.0 gseapy==1.1.1 GEOparse==2.0.4

# Clean for puny space savings
RUN apt-get clean -y && apt-get autoremove -y
