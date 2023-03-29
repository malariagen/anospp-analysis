# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['anospp_analysis']

package_data = \
{'': ['*']}

install_requires = \
['BioPython', 'cutadapt', 'numpy', 'pandas', 'seaborn']

setup_kwargs = {
    'name': 'anospp-analysis',
    'version': '0.1.1',
    'description': 'ANOSPP data analysis',
    'long_description': 'Anopheles species identification with amplicon sequencing - data analysis',
    'author': 'Alex Makunin',
    'author_email': 'am60@sanger.ac.uk',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://github.com/malariagen/anospp-analysis/',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<4.0',
}


setup(**setup_kwargs)

