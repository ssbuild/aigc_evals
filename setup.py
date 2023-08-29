# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/29 10:17
import os.path

from setuptools import setup, find_packages



def parse_deps():
    install_requires = [
    ]
    path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(path, 'requirements.txt'), mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().replace('\r\n', '').replace('\n', '')
        if not line:
            continue
        install_requires.append(line)
    return install_requires

if __name__ == '__main__':
    install_requires = parse_deps()

    setup(
        name='aigc_evals',
        version='0.2.0',
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
    )