from setuptools import setup, find_packages

setup(
    name='docker_manager',
    version='0.1.0',
    author='Ramachandra Vikas Chamarthi',
    author_email='vikas@qblocks.cloud',
    description='A Docker Management API for allocating Docker containers with CPU and GPU resources.',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/vikasqblocks/monster_docker_manager',
    packages=find_packages(),
    install_requires=[
        'fastapi==0.85.0',
        'uvicorn==0.18.3',
        'docker==6.0.0',
        'psutil==5.9.2',
        'nvidia-ml-py3==7.352.0',
        'pydantic==1.10.2'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
)
