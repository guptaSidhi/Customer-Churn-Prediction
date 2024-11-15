from setuptools import find_packages, setup

REQUIREMENTS_FILE = "requirements.txt"
HYPEN_E_DOT = "-e ."

def get_requirements():
    with open(REQUIREMENTS_FILE,"r") as file:
        requirements = file.readlines()
        requirements_list = [requirement.replace("\n","") for requirement in requirements]

        if HYPEN_E_DOT in requirements_list:
            requirements_list.remove(HYPEN_E_DOT)
        
    return requirements_list


setup(
    name = "CustomerChurnPrediction",
    version = "0.0.0.1",
    author = "Sidhi Gupta",
    author_email = "guptasidhi159@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements()
)