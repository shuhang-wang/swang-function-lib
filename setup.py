from setuptools import setup, find_packages

setup(
    name='swang_lib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # list of this package dependencies
    ],
    entry_points={
        'console_scripts': [
            # any scripts you want to be installed
        ],
    },
    # metadata to display on PyPI
    author="Shuhang Wang",
    author_email="swang@cellinobio.com",
    description="Helper functions",
    keywords="Reding",
    url="http://example.com/HelloWorld/",   # project home page, if any
    project_urls={
        "Source Code": "https://github.com/shuhang-wang/swang_lib",
        # any other relevant links
    }
    # could also include long_description, download_url, classifiers, etc.
)
