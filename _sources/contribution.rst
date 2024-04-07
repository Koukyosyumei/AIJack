:orphan:

.. _contribution:

Contribution Guide
==================

Welcome to AIJack's Contribution Guide!

We're thrilled you're interested in contributing to AIJack. This guide outlines the process for submitting code changes and ensuring they adhere to our project's style and formatting conventions.

Getting Started
---------------

Fork the Repository
^^^^^^^^^^^^^^^^^^^

* Head over to the AIJack repository on GitHub (https://github.com/Koukyosyumei/AIJack).
* Click the "Fork" button to create your own copy of the repository.

Clone Your Fork
^^^^^^^^^^^^^^^

* Open your terminal and navigate to your desired local directory.
* Use the git clone command to clone your forked repository:

.. code-block:: bash

    git clone https://github.com/<your-username>/AIJack.git
    # Replace <your-username> with your GitHub username and <project-name> with the actual project name.

Set Up a Development Environment
--------------------------------

* Build and install AIJack from source code

.. code-block:: bash

    cd AIJack

    # install the dependencies
    apt install -y libboost-all-dev
    pip install -U pip
    pip install "pybind11[global]"

    # install the editable version
    pip install -e .


Coding Style and Formatting
---------------------------

Google-Style Docstrings
^^^^^^^^^^^^^^^^^^^^^^^

We use Google-style docstrings to provide clear and consistent documentation for functions, classes, and modules.
Refer to the Google Python Style Guide (https://github.com/google/styleguide/blob/gh-pages/pyguide.md) for detailed formatting instructions.

Black Code Formatter
^^^^^^^^^^^^^^^^^^^^

We utilize Black, a popular code formatter, to maintain consistent code style throughout the project.

Ensure Black is installed (pip install black) in your virtual environment.

To format your code before committing, run:

.. code-block:: bash

    black .


Isort Import Organizer
^^^^^^^^^^^^^^^^^^^^^^

isort helps organize imports in a consistent manner.

Install isort (pip install isort) in your virtual environment.

To organize imports, run:

.. code-block:: bash

    isort .

Making Changes
--------------

Create a Branch
^^^^^^^^^^^^^^^

* Use git checkout -b <branch-name> to create a new branch for your changes. Replace <branch-name> with a descriptive name (e.g., fix-issue-123).

Implement Your Changes
^^^^^^^^^^^^^^^^^^^^^^

* Make your code modifications in the appropriate files.
* Adhere to the coding style and formatting conventions outlined above.

Test Your Changes
^^^^^^^^^^^^^^^^^

* Write unit tests (if applicable) to verify your code's functionality and prevent regressions.

* Run existing tests with pytest to ensure they still pass after your modifications.

Commit Your Changes
^^^^^^^^^^^^^^^^^^^

* Stage your changes using

.. code-block:: bash

	git add <file1> <file2>....

* Commit your staged changes with a descriptive message using

.. code-block:: bash

	git commit -m "<commit message>".

Push Your Changes to Your Fork
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Push your branch to your forked repository on GitHub:

.. code-block:: bash

	git push origin <branch-name>

Submitting a Pull Request
^^^^^^^^^^^^^^^^^^^^^^^^^

* Navigate to Your Fork on GitHub:
* Go to your GitHub repository.
* Create a Pull Request:

  * Click on the "Pull requests" tab.
  * Click on "New pull request" and select the branch containing your changes.
  * Provide a clear and concise title and description for your pull request.
  * Click on "Create pull request" to submit it for review.
  * Code Review and Merging

Project maintainers will review your pull request and provide feedback.
Address any comments or suggestions raised during the review process.
Once your pull request is approved, it will be merged into the main project repository.

Additional Tips
---------------

* Consider running black . and isort . before committing your changes to ensure consistent formatting.
* Provide clear and concise commit messages that describe the purpose of your changes.
* If you're unsure about anything, feel free to ask for help! You can create an issue on the project's GitHub repository.

Thank you for your contribution to AIJack!
