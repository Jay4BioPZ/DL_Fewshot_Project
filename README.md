# CS-502 Project - Fewshot-Benchmark: Differentiable Closed-Form and Iterative Solver

Berta Céspedes Sarrias, Hadi Hammoud, Yung-Cheng Chiang

## Project Description

The project implements the few-shot learning algorithm proposed by **Bertinetto et al., 2018** ([arXiv](https://arxiv.org/abs/1805.08136), [GitHub](https://github.com/bertinetto/r2d2/tree/master?tab=readme-ov-file)) on two biomedical datasets: SwissProt and Tabula Maris. The algorithm is runned under a hydra framework and the result is compared against other few-shot learning benchmark algorithms with the same backbone.

## Implementation

Two algorithm modules were added:

- [`R2D2`](./fewshotbench/methods/r2d2.py): the "Ridge Regression Differentiable Discriminator".
- [`LRD2`](./fewshotbench/methods/lrd2.py): the "Logistic Regression Differentiable Discriminator".

## Instructions

To run the code, let's say training `r2d2` on SwissProt dataset, you can use the following command:

```bash

# Activate the conda environment
conda activate fewshotbench

# Go to the fewshotbench folder
cd DL_Fewshot_Project/fewshotbench/

# Set up experiment name
exp_name="your_experiment_name"

# Run the experiment with selected method and hyperparameters
python run.py exp.name=${exp_name} method=r2d2 dataset=swissprot lr=0.0001 n_way=5 n_shot=5

```

Note that in our implementation, `lr`, `n_way` and `n_shot` are set to be tunable from the configuration file. Thus it is necessary to specify them in the command line.

<!-- ## Environment setup

1. Follow [VM cuda setup tutorial](https://docs.google.com/document/d/1VOyCTOin7JZadlxLMJ457mo7ihypHYT3U2IA83Ba5VY/edit) to get GCP VM instance with miniconda and cuda installed.
2. Establish ssh connection to GCP VM instance as well with local machine.
3. Install git in command line with `sudo apt-get install git`.
4. Set up git user name and email with `git config --global user.name "Your Name"` and `git config --global user.email "
5. Install [github CLI](https://github.com/cli/cli/blob/trunk/docs/install_linux.md) features with

```bash
type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y
```

6. Login to github with `gh auth login`.
7. In the VM, go to the directory you want to clone the repo to, and clone the repo with `gh repo clone Jay4BioPZ/DL_Fewshot_Project`.
8. Go to the project directory `cd .YOURFOLDER/DL_Fewshot_Project`, follow the instruction in [project readme](./fewshotbench/README.md), run `conda env create -f environment.yml`.
9. With VSCode, install the remote-ssh extension. Open a new window, press `Ctrl+Shift+P`, type `Remote-SHH: Connect to Host`, remote connect the VM through `ssh <username>@< VM external ip address>`.
10. You shall now see the external ip address in the bottom left corner of the VSCode window. Further install all recommended extensions (i.e., copilot, python...) so that you can use the VSCode IDE to develop the project.

## Dataset

The dataset should be manually download as instructed in [project readme](./fewshotbench/README.md) and put in the `.YOURFOLDER/fewshotbench/data` folder. They should be ignored by git. -->