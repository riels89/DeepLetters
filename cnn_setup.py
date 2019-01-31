from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES  = ['matplotlib', 'pandas','numpy','cv2','math','sklearn','sys','os','random','tensorflow','logging']
                      
setup(
  name='CNN_Packages',
  version='1',
  author = 'Riley White',
  author_email = 'rileywhite89@gmail.com',
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages(),
  description='Packages required to train DeepLetters CNN')
