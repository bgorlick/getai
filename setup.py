from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='getai',
    version='0.0.5',
    author='Ben Gorlick',
    author_email='ben@unifiedlearning.ai',
    description='GetAI - Asynchronous AI Downloader for models, datasets and tools',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/bgorlick/getai',
    packages=find_packages(),
    install_requires=[
        'aiohttp>=3.7.4,<4',
        'aiofiles',
        'prompt_toolkit',
        'rainbow-tqdm',
        'PyYAML',
    ],
    package_data={
        'getai': ['config.yaml'],
    },
    extras_require={
        'huggingface': [],
    },
    entry_points={
        'console_scripts': [
            'getai=getai.__main__:run',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
