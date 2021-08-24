from distutils.core import setup
setup(
  name = 'visuEL',         # How you named your package folder (MyLib)
  packages = ['visuEL'],   # Chose the same as "name"
  version = '0.28',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'visuEL - VISualization of Event Logs: contains an event logs sampler and a visualizer so that thousands of traces from an event logs can be easily visualized using few representatives traces on a SVG',   # Give a short description about your library
  author = 'Gaël Bernard',                   # Type in your name
  author_email = 'gael.bernard@utoronto.ca',      # Type in your E-Mail
  url = 'https://github.com/gaelbernard/visuEL/',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/gaelbernard/visuEL/archive/refs/tags/v_01.tar.gz',    # I explain this later on
  keywords = ['process mining', 'event logs', 'visualizer', 'sampler'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'pandas',
          'scikit-learn',
          'py2opt',
          'random2',
          'editdistance',
          'matplotlib',
          'svgwrite',
          'pm4py',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Science/Research',      # Define that your audience are developers
    'Topic :: Software Development',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)