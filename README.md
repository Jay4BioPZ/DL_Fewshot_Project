# DL_Fewshot_Project

CS-502 fewshot learning project

## Environment setup

1. Follow [VM cuda setup tutorial](https://docs.google.com/document/d/1VOyCTOin7JZadlxLMJ457mo7ihypHYT3U2IA83Ba5VY/edit) to get GCP VM instance with miniconda and cuda installed.
2. Establish ssh connection to GCP VM instance as well with local machine.
3. With VSCode, install the remote-ssh extension. Open a new window, connect the VM through `ssh <username>@< VM external ip address>`.
4. Install git with `sudo apt-get install git`.
5. Set up git user name and email with `git config --global user.name "Your Name"` and `git config --global user.email "
6. Install [github CLI](https://github.com/cli/cli/blob/trunk/docs/install_linux.md) features with

```bash
type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y
```

7. Login to github with `gh auth login`.
8. In the VM, go to the directory you want to clone the repo to, and clone the repo with `gh repo clone Jay4BioPZ/DL_Fewshot_Project`.
9. Go to the project directory `cd .YOURFOLDER/DL_Fewshot_Project`, follow the instruction in [project readme](./fewshotbench/README.md), run `conda env create -f environment.yml`.