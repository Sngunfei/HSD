from setuptools import setup

setup(
    name='GraphEmbedding',
    version='0.1',
    description='Techniques for Structurally Graph Embedding.',
    author='Song Yunfei',
    author_email='syfnico@foxmail.com',
    packages=['ge'],
    install_requires=['torch', 'numpy', 'gensim', 'networkx', 'tqdm', 'joblib', 'matplotlib', 'scikit-learn',
                      'scipy', 'tensorflow', 'keras', 'pandas', 'fastdtw', 'pyecharts', 'pygsp', 'texttable',
                      'pymongo', 'torch']
)
