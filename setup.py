import pathlib

import setuptools


_here = pathlib.Path(__file__).resolve().parent

name = "ModeRL"
author = "Aidan Scannell"
author_email = "scannell.aidan@gmail.com"
description = (
    "Mode constrained model-based-reinforcement learning in TensorFlow/GPflow."
)

with open(_here / "README.md", "r") as f:
    readme = f.read()

url = "https://github.com/aidanscannell/" + name

license = "Apache-2.0"

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Intended Audience :: Information Technology",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
]
keywords = [
    "model-based-reinforcement-learning",
    "mixtures-of-gaussian-process-experts",
    "gaussian-processes",
    "machine-learning",
    "variational-inference",
    "bayesian-inference",
    "constrained-reinforcement-learning",
    "planning",
]

python_requires = "~=3.7"

install_requires = [
    "gpflow>=2.6.1",
    "numpy",
    "tensorflow-probability",
    "tensorflow>=2.4.0; platform_system!='Darwin' or platform_machine!='arm64'",
    "tensorflow-macos>=2.4.0; platform_system=='Darwin' and platform_machine=='arm64'",
    "tensor_annotations==1.0.2",
    f"mosvgpe @ file://{_here}/subtrees/mosvgpe",
    f"simenvs @ file://{_here}/subtrees/simenvs",
    "wandb",
]
extras_require = {
    "experiments": ["hydra-core", "palettable", "tikzplotlib"],
    "examples": ["jupyter", "hydra-core"],
    "dev": ["black[jupyter]", "pre-commit", "pyright", "isort", "pyflakes", "pytest"],
}

setuptools.setup(
    name=name,
    version="0.1.0",
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    keywords=keywords,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=url,
    license=license,
    classifiers=classifiers,
    zip_safe=False,
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    packages=setuptools.find_packages(exclude=["examples"]),
)
