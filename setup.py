from setuptools import find_packages, setup
from typing import List

requirement_lst = []

def get_requirements():
    """ This function will return a list of requirements"""
    try:
        with open("requirements.txt", 'r') as file:
            lines = file.readlines()

            for line in lines:
                requirement = line.strip()

                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")
    
    return requirement_lst

setup(
    name = 'Sentiment_Analysis_Review',
    version = '0.0.1',
    author = 'Olamide Bamigbola',
    author_email = 'abdullhateef@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements()
)
