from setuptools import setup, find_packages

setup(
    name='pinkrigs_tools',
    version='0.1.0',
    description='Python module for querying and formatting PinkRigs dataset',
    author='Flora Takacs',
    author_email='flora.takacs.15@ucl.ac.uk',
    url='https://github.com/takacsflora/floras-helpers',  
    packages=find_packages(include = ['pinkrigs_tools', 'pinkrigs_tools.*']), 
    install_requires=[
        "floras_helpers @ git+https://github.com/takacsflora/floras-helpers.git@main#egg=floras_helpers", 
        'ipykernel',
        'pyarrow==8.0.0', # for parquetread support in pd
        'pandas==1.3.5',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',    

)
