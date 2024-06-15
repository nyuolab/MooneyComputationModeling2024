# Nature2023MooneyScripts
## Download other required data to reproduce our results
To produce results, go our OSF page, download and decompress the `data.zip` file and the model checkpoint `model_checkpoint.tar`. Put the contents of data at the root, and create a folder `results/checkpoints` and put the checkpoints there. Then you can use `snakemake` to reproduce any results you want.

## System requirements
This code has been tested on MacOS Sonoma and big sur, and on a slurm-based linux HPC cluster. It's recommended you have GPUs or high memory CPUs for most of the operations in our code. We used `python 3.11` for our programs. In principle other versions could work, but be warned that this is not teseted.

## Python requirements
We recommend using a conda environment to reproduce the results. First, create a conda environment: 
```bash
conda create -n mooney_comp python=3.11
```

Then, install the requirements:
```bash
pip install -r requirements.txt
```

This should run without errors.

## Using `snakemake`
Snakemake is a great tool for automating workflows of scripts with complicated interactions. We use `snakemake` to manage the processes of training models and analyzing models. We will only walk through how to reproduce the results. To learn more about how it works, refer to [the official snakemake tutorials](https://snakemake.readthedocs.io/en/stable/tutorial/tutorial.html).

### Config files
`config.yaml` contains experiment-related configurations, and `config_debug.yaml` is a reduced set used to debug.

### Submitting jobs automatically using `snakemake`
Snakemake provides native slurm support, which you can use to automatically submit jobs on a slurm-managed HPC. When you want to execute a certain rule or want to obtain a certain output file, simply run:
```bash
snakemake {wanted_file_name} --configfile config.yaml --profile cluster_config
```

`config.yaml` contains information about the analyses and model training itself, whereas `cluster_config` contains information like slurm submission script, and how much resource + which partition to allocate to each job. You will need to heavily modify this file according to your need if you want to let `snakemake` submit its own jobs.

## What do I need to run in order to reproduce the figures?
First run `snakemake {figure you want} --configfile config.yaml` to obtain the data required. Let's say you want to produce figure 5b, you can run: ```snakemake fig_5b ...` to produce the data necessary to produce it. Then you can run the corresponding notebooks to produce the figures.

## Training your own model
Remember to replace the path of ImageNet-1k with your own directory, and run `snakemake train_all_models`. Note that depending on the number of processes you provide to `snakemake`, you might be training more than one job at the same time.