# Contributing

Contributions are welcome and are greatly appreciated! Every little chunk helps, 
and credit will always be given where due.

There are many ways to contribute. The following section lists only a few.

## Types of Contributions

###### Report Bugs

~~~~~~~~~~~
Report bugs at https://github.com/visionjo/pykinship/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/visionjo/pykinship/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `pykinship` for local development.

1. Fork the `pykinship` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/pykinship.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv pykinship
    $ cd pykinship/
    $ python setup.py develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the tests, including testing other Python versions with tox::

    $ flake8 pykinship tests
    $ python setup.py test or py.test
    $ tox

   To get flake8 and tox, just pip install them into your virtualenv.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request (PR), check the following criteria is met:

    1. The PR should include tests.
    2. If the PR adds functionality, the docs should be updated. Put your new 
    functionality into a function with a docstring, and add the feature to the list 
    in README.rst.
    3. The PR should work for Python 3.5, 3.6, 3.7 and 3.8, and for PyPy. Check
       https://travis-ci.org/jvision/pykin/pull_requests
       and make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

    $ python -m unittest tests.test_face_recognition

Specific Ways to Contribute
----
There are too many items to list. In fact, there are too many lists to list. In 
other words, whether it be research-based, database-related, features or demos, 
project-level components, interactive image gallery, dashboard for automatic 
scoring of test results (i.e., as labels are still private, but we score upon 
request), or even as a means to learn, a means to extend your research to 
kin-based visual problems; whether you are purely CS, or entirely not (e.g., 
anthropologist)-- there is something within for just about anyone. 

If there is any way we can be of assistance, either personal, unfunded project, 
or as part of a larger project, where a more formal collaboration is desired, 
do, please, just let me know.

Few areas in need of man-power:

    1. Unit tests, along with complete build package at project-level.
    2. Exploratory data analysis demos, for both gained insights and demo for handling face data.
    3. Add models and extend benchmarks with other methods.
    4. Incorporate demos that use one of the more popular API (e.g., pytorch, keras, tf, such)
    5. Many other way; furthermore, if there is something that comes to your mind, chances are it will be of our interest. Thus, do not hesitate to reach out with any and all comments or questions.
    
    
---
Author: ***J. Robinson***

Created on ***29 May 2020***