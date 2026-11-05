"""Setup for the ToR-OT package (``torot/``). The baselines run from the checkout with their own environments
(``baselines/README.md``)."""

import setuptools


def _parse_requirements(path: str) -> list[str]:
    with open(path) as fp:
        lines = [line.split("#")[0].strip() for line in fp.read().splitlines()]
    return [line for line in lines if line]


setuptools.setup(
    name="torot",
    version="2.0.0",
    description="ToR-OT: Learning Optimal Transport Plans Via Autoregressive Token Regression",
    author="Ivan Jaen Marquez, Sotirios Panagiotis Chytas, Vikas Singh (University of Wisconsin-Madison)",
    author_email="ivanjaenm@cs.wisc.edu",
    python_requires=">=3.11",
    install_requires=_parse_requirements("torot/requirements.txt"),
    packages=setuptools.find_packages(exclude=("torot.tests", "baselines", "baselines.*")),   # baselines run from the checkout
    package_data={"torot": ["configs/*.yaml"]},
)
