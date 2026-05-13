# Scaling Up CPUs and GPUs Inside Your Analysis Pipelines

In this tutorial, you'll learn:

* How to use a container with a Jupyter Notebook on Sherlock
* How to use the `dask` package in Python to scale complex analyses over multiple CPUs
* How to use `DistributedDataParallel` (DDP) in PyTorch to distribute across multiple GPUs


## Today's Tutorial Agenda

* Installing containers and Jupyter kernels from a shared folder [[link]()]
* Introduction to Dask for parallel computing [[link]()]
* Distributing to multiple CPUs in Dask [[link]()]
* Integrating Dask with your analysis scripts [[link]()]
* Loading your model onto a single GPU in PyTorch [[link]()]
* Using `DistributedDataParallel` to distribute to multiple GPUs with PyTorch [[link]()]

## How to Install a Custom Jupyter Lab Kernel from a Containerized Environment
Installing Python packages with specific versions and dependencies can be challenging in an HPC workspace like Sherlock. One potential solution is to containerize your coding environment (See our [workshop on package management](https://github.com/stanford-sdss/package-management/) to learn how to build your own containers!). We took this approach and containerized our custom Python environment which you can use to access all the Python packages used in this tutorial from an Open On Demand Jupyter Lab session using the following steps. 

1. Use `git` to clone this repository onto Sherlock and cd into it.
3. Run `chmod +x update-kernel.sh` from the command line.
4. Next, run `source update-kernel.sh` from the command line.
5. Verify that your new kernels, `dask-distribute` and `pytorch-ddp`, exist by running `ls -la ~/.local/share/jupyter/kernels/`.

**Note:** The `update-kernel.sh` script works on Sherlock, because it automatically installs containers from a shared folder, `/scratch/groups/jfreshwa/containers/`. In order to expose this folder to your Sherlock session, you'll need to run `ls /scratch/groups/jfreshwa/containers/` before your kernels will work. Additionally, if you are attempting this off of Sherlock, you'll need to download the containers from the MATRICS Bootcamp GitHub.

### Do you have any questions? 
Please reach out to us at [matrics-bootcamp@stanford.edu](mailto:matrics-bootcamp@stanford.edu) or set up a General Consultation with us at [this link](http://sdss-compute-consultation.stanford.edu/).

### Would you like to provide feedback?
Please provide anonymous feedback [here](https://forms.gle/x3wB8qMPWBbeNosR9).