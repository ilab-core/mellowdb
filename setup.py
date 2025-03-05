from setuptools import find_packages, setup

setup(
    name='mellow-db',
    version='0.1.0',
    description='A vectordb solution with metadata filtering',
    author='Merve Kantarci, iLAB R&D',
    author_email='mkantarci@ilab.com.tr',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    extras_require={
        'server': [
            'python-dotenv',
            'faiss-cpu==1.9.0',
            'sqlalchemy==2.0.36',
            'scikit-learn>=1.5.2',
        ],
        'pytest': [
            'pandas>=2.2.3',
            'pytest>=8.3.3',
            'python-dotenv',
            'scikit-learn>=1.5.2',
            'pre-commit==2.15.0',
        ],
    },
    python_requires='>=3.8',
    include_package_data=True
)
