import yaml
from tqdm import tqdm
import os
import glob

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config("config.yaml")
packages_config = config["packages"]
PROJECT_DIR = config["project_dir"]

run_list = [k for k in config["main"].keys() if config["main"][k] is True]

def reset_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        delfiles = glob.glob(path+"/*")
        for f in delfiles:
            if os.path.isfile(f):
                os.remove(f)

def chdir_config(config):
    for key in config:
        if isinstance(config[key], dict):
            config[key] = chdir_config(config[key])
        elif "path" in key.lower():
            config[key] = os.path.join(PROJECT_DIR, config[key])
            folder = config[key]
            if len(os.path.splitext(config[key])) > 0:
                folder = os.path.dirname(config[key])
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            if "output" in key.lower():
                reset_directory(folder)
            
    return config

config["packages"] = chdir_config(config["packages"])

print("Copying config files to package directories..")

for package in run_list:
    print(f"Copying to {package}")
    cfg = config["packages"][package]
    package_dir = os.path.join(PROJECT_DIR, package)
    with open(os.path.join(package_dir, 'config.yaml'), 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)


for package in run_list:
    print(f"Executing {package}..")
    program_path = os.path.join(PROJECT_DIR, package,"main.py")
    target_directory = os.path.dirname(os.path.abspath(program_path))
    print(target_directory)
    print(program_path)
    

with open("run.sh", "w") as file:
    file.write(f"#!/bin/bash\n")
    for package in run_list:
        program_path = os.path.join(PROJECT_DIR, package,"main.py")
        file.write(f"cd {os.path.dirname(program_path)}\n")
        file.write(f"python3 main.py\n")
    file.write(f"echo 'Execution Completed!'")

print("script written to run.sh")