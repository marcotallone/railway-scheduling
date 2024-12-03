from setuptools import setup, find_packages

setup(
    name='railway',
    version='1.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    author='Marco Tallone',
    author_email='marcotallone85@gmail.com',
    description='Railway Maintenance Optimization',
    long_description="""
    Module to model the optimal scheduling of railway maintenance projects 
    in a railway network.""",
)

# Main to explain usage in case --help is used
if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--help' or sys.argv[1] == '-h':
        print("""To install this package, run the following command:\tpip install -e .""")

    if len(sys.argv) > 1 and sys.argv[1] == '--usage':
        print("""To install this package, run the following command:\tpip install -e .""")