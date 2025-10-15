
"""Setup for pip package."""
import setuptools

def _strip_comments_from_line(s: str) -> str:
  """Parses a line of a requirements.txt file."""
  requirement, *_ = s.split('#')
  return requirement.strip()


def _parse_requirements(requirements_txt_path: str) -> list[str]:
  """Returns a list of dependencies for setup() from requirements.txt."""

  with open(requirements_txt_path) as fp:
    # Parse comments.
    lines = [_strip_comments_from_line(line) for line in fp.read().splitlines()]
    # Remove empty lines and direct github repos (not allowed in setup.py)
    return [l for l in lines if (l and 'github.com' not in l)]


setuptools.setup(
    name='TF4OT',
    version='1.0',
    description='TF4OT: Transformers for Optimal Transport',
    author='Ivan Jaen Marquez @ UW-Madison, Takis Chytas @ UW-Madison, Vikas Singh @ UW-Madison',
    author_email='ivanjaenm@gmail.com',
    install_requires=_parse_requirements('requirements.txt'),
    packages=setuptools.find_packages(),
    package_data={
        '': ['**/*.gin', '**/*.proto'],
    })
