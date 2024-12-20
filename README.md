# Coabundance networks of NSCLC lung tumor cells from scRNAseq data

The main objective of this project is to perform an ecological analysis of the cell types in NSCLC lung tumor tissues from scRNA-seq data, in both early and late tumor stages.

The chosen methodology is to annotate cell types with the help of a newly trained reference atlas. For this, we use the [scvi](https://github.com/scverse/scvi-tools) framework. We base our work on the study by [Salcher, Sturm, Horvath et al. 2022](https://pubmed.ncbi.nlm.nih.gov/36368318/), where they gathered and standardized most of the data. 

Additionally, we expanded the atlas with four more datasets, optimized the hyperparameters of our deep learning model, and trained it to also predict tumor cell types. The prediction was better than with the most used metho [InferCNV](https://github.com/broadinstitute/infercnv)

Subsequently, we obtained coabundance metrics. We chose the mutual inference (MI) metric, as it captures non-linear correlations in the data, and the methodology in the [ARACNE-AP](https://github.com/califano-lab/ARACNe-AP) package ensures a smaller possibility of spurious correlations.

After that, we obtained various visualizations of the networks and extracted mesoscopic and global properties of the graphs.

The workflow for getting from the raw AnnData files to the coabundance graphs and their analysis can be seen in the figure below:

![ScRNAseq Workflow](Workflow_dark.png "scRNA-seq Workflow")

## Folder structure

The workflow is spread across various notebooks, where one can see the figures generated in the proccess and some short explanations. 
We plan to automatize the process with Nextflow, like they do in the repo of the [scLUCA](https://github.com/icbi-lab/luca) project.

But for now al the necessary steps are contained in the notebooks and scripts, though a little scrambled.

### Notebooks

- *get_data.ipynb*: Contains the code for exploring and downloading all possible datasets to extend the scLUCA atlas. It was important to have UMI counts and tumor stage annotation.
- *graph_builder.ipynb*: A mashup of too many things that need to be separated. It contains the crucial parts of gathering all annotations, running the coabundance analysis, and plotting the networks.
- *nb_filter*: Filters the cells and genes by predefined quality control metrics.
- *nb_annot*: Annotates tissue from every study with Lung Atlas reference maps. Only healthy cells.
- *nb_annotRefatlas*: Annotates tissue from a new study by doing surgery and has integrated quality plots. Also annotates broad tumor cell types. Includes a notebook for annotation with label transfer via neighbors, but it had worse outcomes (`nbLabelPreds`). Ther eis also a nb that transfers the newly created clusters to another dataset (`extendPreds_{dataset}.ipynb`).
- *nb_ikarus*: Runs the ikarus prediction on every sample. A prediction that uses logistic regression and network projection to predict tumor cells.
- *nb_infercnv*: Runs InferCNV on every sample. This infers from transcripts, places in the chromosomes where there should be copy number variations.
- *nb_DE_wilcox*: Extracts marker genes of clusters from existing cell annotations with the Wilcox method. It also enriches for Hallmark gene ontologies. Its a bit bit convoluted and doesnt consider batch effects, but only because it scanpy doenst consider abundance of cell types. The notebook `DE_param` is incomplete and attempts to do pseudo-bulk differential expression with MAST. There is also a `modal_DE.py` script to run the Wilcox marker gene extraction in modal. It generalizes well for all datasets, but one has to upload the files to the volume manually for now.
- *nb_DE_SCT*: Extracts marker genes of clusters from existing cell annotations with the GLM method `SCTransform v2`. It also corrects for batch effects per sample and enriches for Hallmark gene ontologies. This method has a parameter estimation method that corrects for lowly expressed genes and is much faster than the `lvm_DE` method, which takes advantage of our dimensional reduction VAEs with scANVI. The `test_de.py` script is to be run in the lambda function service Modal, as it requires a lot of GPU RAM and takes around 35 minutes for 3 samples. It uses the `lvm_DE` method mentioned above.
- *nb_tumorUMAP*: Notebook to check the tumor predictions. It has the DE part integrated. `Tumor_Annot.ipynb` contains explanations of the methods used.
- *nb_refAtlas*: Contains the notebook `vay_raytune` for running and inspecting various experiments of hyperparameter exploration. The notebook `scANVImodel` has the reasoning and training of the actual model. It also includes `Tumor_subcluster.ipynb` that redefines the atlas with new unsupervised tumor cells. Some attempts to accelerate this with `faiss` are in the python scripts starting with faiss.
- *nb_graphAnalysis*: Parametrized file for doing a MI (Mutual Inference) graph analysis of an already provided list of edges between cell types.

### Misc

- *outputARACNE*: Contains all the files for the generation and output of the networks by ARACNE, including functional enrichment.
- *metadata*: Contains information about the studies used and data about the groups.
- *utils*: Contains custom plotting and analysis functions.

## Running

We have designed a Docker image that has all the necessary libraries. It is, however, very large because it includes all the R and Python packages, including those for ML. It is around 15GB without InferCNV and Ikarus.

To run the notebook, you should run the container with this notebook repository mounted as a volume. In the following command, `$HOME/2021-SC-HCA-LATAM/CONTAINER` is the path of the repository, and the other path is where the large data files would be stored.

You must install the apt package `docker-nvidia` for the GPU flags to work and, of course, have a working CUDA installation.

```bash
docker run --interactive --runtime=nvidia --gpus all --tty --name comp_onco --publish 8888-8892:8888-8892 --volume $HOME/2021-SC-HCA-LATAM/CONTAINER:/root/host_home --volume /datos:/root/datos --workdir /root/host_home/ netopaas/comp-onco:paga /bin/bash
```

### Jupyter lab

We publish some ports to use the Jupyter server.

After that, just run the command `jl` inside the container, and a Jupyter Lab server will be launched.

To acces it from a local computer you can do some port forwarding with something akin to this command:

```bash
ssh -p 5265 -N -f -L 5432:localhost:8888 sefirot.inmegen.gob.mx
```

Where 5432 is the localport where you will be running the server. 8888 is the port in the remote machine where juptyer server is being forwarded to by the container.
And -p 5265 is the por tof the remote host. You have to have and ssh-agent with your user credentials for this to work. We recommend it as it frees you of the hassle
of inputting a password everyitme.

### VS Code

You can also skip the port forwarding and work in a more integrated environment, by using the remote explorer and dev contianers feature of VScode.

Just add a new host for the remote host via shh and enter it. VScode will automatically install all the necessary packages to be able to interact with the host.
After that go to the Dev Containers tab in the Remote Explorer tab and choose the container that you ran as per the instructions above.

When opening a notebook, we recommend attaching to a running jupyter server, which is launched with the `jl` command in a shell in the container, like above.
This avoids losing the running kernel if you close the window you are working on.

## Troubleshooting

Due to the long training and annealing times, Jupyter Lab sometimes cannot connect. Use this to get a console to the kernel:
```bash
jupyter console --existing /root/.local/share/jupyter/runtime/kernel-9ff04919-e8c2-4ecf-92ce-c66b988720e5.json
```
For this you need to have `pip install jupyter-console` installed.

This could be easier in a newer version of Jupyter Lab. To locate the corresponding JSON file, you can use `htop` with the option to not display user branches and see the memory it is using.

In this image: `python:3.11.4`. It is important to install the Docker NVIDIA package to transfer your CUDA installation to the containers.

If you keep your repository inside the container via volumes, the user might change. We recommend configuring the SSH keys inside the container. These are the steps:

```bash
git config --global user.email ernesto.paas@ciencias.unam.mx
git config --global user.name "Ernesto Paas"
```

We suggest saving a key pair that can be generated with the command `ssh-keygen -t ed25519 -C "your_email@example.com"` in the Docker volume (folder) that contains the repository, and then copying them to `~/.ssh/id_ed25519` etc., to have SSH authentication with GitHub. Remember to have the SSH agent activated `eval "$(ssh-agent -s)"` and the key added `ssh-add ~/.ssh/id_ed25519`. Ensure that the key is read-only.
