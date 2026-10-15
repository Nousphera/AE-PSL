"""Install AE-PSL directly from this source checkout."""
from pathlib import Path

from setuptools import find_packages, setup

root = Path(__file__).resolve().parent
requirements = [line.strip() for line in (root / 'requirements.txt').read_text().splitlines()
                if line.strip() and not line.lstrip().startswith(('#', '-e'))]
setup(
    name='ae-psl',
    version='0.2.0',
    description='AutoEncoder-Compressed Parallel Split Learning for Pre-Trained Model Fine-Tuning',
    python_requires='>=3.12,<3.14',
    license='MIT',
    license_files=['LICENSE'],
    package_dir={'': 'src'},
    packages=find_packages('src'),
    install_requires=requirements,
    entry_points={'console_scripts': [
        'ae-psl=ae_psl.experiment:main',
        'ae-psl-data=ae_psl.prepare_data:main',
        'ae-psl-smoke=ae_psl.smoke:main',
    ]},
)
