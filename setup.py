"""Setup for the ToR-OT package (v2 in ``torot/``; the legacy v1 code in ``TF4OT/`` is
installed alongside but its pinned dependencies are listed in ``TF4OT/requirements.txt``)."""

import setuptools


def _parse_requirements(path: str) -> list[str]:
    with open(path) as fp:
        lines = [line.split("#")[0].strip() for line in fp.read().splitlines()]
    return [line for line in lines if line]


setuptools.setup(
    name="torot",
    version="2.0.0",
    description="ToR-OT: Learning Optimal Transport Plans Via Autoregressive Token Regression",
    author="Ivan Jaen Marquez, Takis Chytas, Vikas Singh (University of Wisconsin-Madison)",
    author_email="ivanjaenm@cs.wisc.edu",
    python_requires=">=3.11",
    install_requires=_parse_requirements("torot/requirements.txt"),
    packages=setuptools.find_packages(exclude=("torot.tests", "baselines", "baselines.*")),   # baselines run from the checkout
    package_data={"torot": ["configs/*.yaml"]},
)
