from setuptools import setup, find_packages

install_requires = [
    'numpy>=1.10.0',
    'scipy',
    'pandas',
    ]

setup(name='Shifted Beta-Geometric Survival Analysis',
      version='0.1',
      description='An alternative to survival analysis using '
                  'the shifted beta-geometric model.',
      classifiers=[
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.5",
          ],
      author='Fernando Nogueira',
      author_email='fmfnogueira@gmail.com',
      url='',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      )
