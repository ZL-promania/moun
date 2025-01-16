from setuptools import setup, find_packages

setup(
    name='veto_eff',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # 在此添加你的项目依赖库，如 numpy, pandas 等
        'numpy',
        'pandas',
    ],

)