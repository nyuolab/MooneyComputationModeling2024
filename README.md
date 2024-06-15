# Nature2023MooneyScripts
To produce results, go our OSF page, download and decompress the `data.zip` file and the model checkpoint `model_checkpoint.tar`. Put the contents of data at the root, and create a folder `results/checkpoints` and put the checkpoints there. Then you can use `snakemake` to reproduce any results you want.

To retrain model, remove existing checkpoints and run `snakemake train_models`.
