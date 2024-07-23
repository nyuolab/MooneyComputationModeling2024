#!/bin/bash
# properties = {properties}

source ~/.bashrc
conda activate mooney_comp
exp_pypath mooney_comp

{exec_job}
