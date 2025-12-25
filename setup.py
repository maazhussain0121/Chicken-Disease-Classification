import setuptools

with open("README.md", "r", encoding="utf-8") as fh:    
    long_description = fh.read()


__version__ = "0.0.0"

REPO_NAME = "Chicken-Disease-Classification"
AUTHOR_USER_NAME = "Maaz Hussain"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "maazhussain0121@gmail.com"


setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN based image classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"http://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    projecrt_urls={
        "Bug Tracker": f"http://github.com?{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)