# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/29 10:17
import os.path

from setuptools import setup, find_packages



def parse_deps():
    install_requires = [
        "mypy",
        "openai >= 0.27.2",
        "tiktoken",
        "blobfile",
        "backoff",
        "numpy",
        "snowflake-connector-python[pandas]",
        "pandas",
        "datasets",
        "fire",
        "pydantic",
        "tqdm",
        "nltk",
        "filelock",
        "mock",
        "langdetect",
        'termcolor',
        "lz4",
        "pyzstd",
        "pyyaml",
        "sacrebleu",
        "matplotlib",
        "pytest",
        "setuptools_scm",
        "langchain",
        "types-PyYAML",
    ]

    return install_requires

if __name__ == '__main__':
    install_requires = parse_deps()

    oaieval = "evals.cli.oaieval:main"
    oaievalset = "evals.cli.oaievalset:main"

    setup(
        name='aigc_evals',
        version='0.0.1',
        description='aigc_evals',
        long_description='torch_training: https://github.com/ssbuild/aigc_evals.git',
        license='Apache License 2.0',
        url='https://github.com/ssbuild/aigc_evals',
        author='ssbuild',
        author_email='9727464@qq.com',
        install_requires=install_requires,
        package_dir={"": "src"},
        packages=find_packages("src"),
        include_package_data=True,
        package_data={"": ["**/*.yaml","**/*.jsonp"]},
        entry_points={
            'console_scripts': [
                'sh_aigc_evals = aigc_evals.cli.sh_aigc_eval:main',

            ],
        }

    )