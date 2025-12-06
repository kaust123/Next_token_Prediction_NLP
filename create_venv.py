#!/usr/bin/env python3
"""create_venv.py
Utility script to create a virtual environment and optionally install requirements.

Usage examples:
  python create_venv.py                # creates .venv
  python create_venv.py --path env     # creates 'env'
  python create_venv.py --install-requirements

On Windows PowerShell, activate with:
  .\<env>\Scripts\Activate.ps1
"""

import argparse
import os
import subprocess
import sys
import venv


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def create_env(path: str, install_requirements: bool, requirements_file: str, upgrade_tools: bool):
    path = os.path.abspath(path)

    if os.path.exists(path):
        print(f"Virtual environment path '{path}' already exists. Leaving it in place.")
    else:
        print(f"Creating virtual environment at: {path}")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(path)
        print("Virtual environment created.")

    # Determine pip executable inside venv
    if os.name == "nt":
        pip_exe = os.path.join(path, "Scripts", "pip.exe")
    else:
        pip_exe = os.path.join(path, "bin", "pip")

    if upgrade_tools and os.path.exists(pip_exe):
        print("Upgrading pip, setuptools, and wheel inside the venv...")
        run([pip_exe, "install", "--upgrade", "pip", "setuptools", "wheel"])

    if install_requirements:
        if os.path.exists(requirements_file):
            print(f"Installing packages from '{requirements_file}' into the venv...")
            run([pip_exe, "install", "-r", requirements_file])
        else:
            print(f"Requirements file '{requirements_file}' not found. Skipping install.")

    print("\nActivation instructions:")
    rel = os.path.relpath(path)
    if os.name == "nt":
        print(f"PowerShell: .\\{rel}\\Scripts\\Activate.ps1")
        print(f"CMD: {rel}\\Scripts\\activate.bat")
    else:
        print(f"bash/zsh: source {rel}/bin/activate")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Create a virtual environment and optionally install requirements.")
    parser.add_argument("--path", default=".venv", help="Path to create the virtual environment (default: .venv)")
    parser.add_argument("--install-requirements", action="store_true", help="Install packages from requirements file after creating the venv")
    parser.add_argument("--requirements-file", default="requirements.txt", help="Requirements file path (default: requirements.txt)")
    parser.add_argument("--no-upgrade-tools", dest="upgrade_tools", action="store_false", help="Do not upgrade pip/setuptools/wheel in the venv")

    args = parser.parse_args()

    create_env(args.path, args.install_requirements, args.requirements_file, args.upgrade_tools)


if __name__ == "__main__":
    main()
