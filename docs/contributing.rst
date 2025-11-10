============
Contributing
============

Thank you for your interest in contributing to COLA! We welcome contributions of all kinds.

Getting Started
===============

1. **Fork the repository**

   Visit https://github.com/understanding-ml/COLA and click "Fork"

2. **Clone your fork**

   .. code-block:: bash

       git clone https://github.com/your-username/COLA.git
       cd COLA

3. **Install in development mode**

   .. code-block:: bash

       pip install -e .
       pip install -r requirements.txt

Types of Contributions
======================

Bug Reports
-----------

Use the `GitHub issue tracker <https://github.com/understanding-ml/COLA/issues>`_.

Include:

- Python version and OS
- COLA version (``import xai_cola; print(xai_cola.__version__)``)
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback

Documentation
-------------

Help improve documentation by:

- Fixing typos or unclear explanations
- Adding examples or tutorials
- Improving docstrings
- Translating documentation

New Features
------------

Before implementing a new feature:

1. Open an issue to discuss the feature
2. Ensure it aligns with project goals
3. Get feedback from maintainers

Then:

- Implement the feature
- Add tests
- Update documentation
- Submit a pull request

Code Quality
------------

Contributions for:

- Refactoring for better performance
- Adding type hints
- Improving test coverage
- Code cleanup

Development Workflow
====================

1. Create a Branch
------------------

.. code-block:: bash

    git checkout -b feature/your-feature-name

2. Make Changes
---------------

Follow the coding style:

- Use 4 spaces for indentation
- Follow PEP 8
- Add docstrings to all public functions
- Write meaningful commit messages

3. Add Tests
------------

.. code-block:: bash

    # Run tests
    pytest tests/

    # Check coverage
    pytest --cov=xai_cola tests/

4. Update Documentation
-----------------------

If you changed the API:

.. code-block:: bash

    cd docs
    make html

    # Check the generated docs
    open _build/html/index.html

5. Commit Changes
-----------------

.. code-block:: bash

    git add .
    git commit -m "feat: add new feature X"

Use conventional commits:

- ``feat:`` New feature
- ``fix:`` Bug fix
- ``docs:`` Documentation changes
- ``test:`` Adding tests
- ``refactor:`` Code refactoring

6. Push and Create Pull Request
--------------------------------

.. code-block:: bash

    git push origin feature/your-feature-name

Then create a PR on GitHub.

Code Style
==========

Python Style
------------

Follow PEP 8:

.. code-block:: python

    # Good
    def calculate_similarity(x, y):
        """Calculate similarity between x and y."""
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    # Bad
    def calc_sim(x,y):
        return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

Docstring Style
---------------

Use NumPy style docstrings:

.. code-block:: python

    def refine_counterfactuals(
        self,
        limited_actions: int,
        features_to_vary: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Refine counterfactuals with limited actions.

        Parameters
        ----------
        limited_actions : int
            Maximum number of feature changes allowed.
        features_to_vary : list of str, optional
            Features that can be modified. If None, all features can vary.

        Returns
        -------
        pd.DataFrame
            Refined counterfactual instances.

        Raises
        ------
        ValueError
            If limited_actions is less than 1.

        Examples
        --------
        >>> cola = COLA(data=data, ml_model=model)
        >>> cola.set_policy(matcher='ot', attributor='pshap')
        >>> refined = cola.refine_counterfactuals(limited_actions=5)
        """
        pass

Testing
=======

Writing Tests
-------------

Add tests for all new features:

.. code-block:: python

    # tests/test_cola.py
    import pytest
    from xai_cola import COLA

    def test_refine_counterfactuals():
        """Test basic refinement functionality."""
        # Setup
        cola = COLA(data=test_data, ml_model=test_model)
        cola.set_policy(matcher='ot', attributor='pshap')

        # Execute
        refined = cola.refine_counterfactuals(limited_actions=5)

        # Assert
        assert len(refined) > 0
        assert refined.shape[1] == test_data.factual_df.shape[1]

Running Tests
-------------

.. code-block:: bash

    # Run all tests
    pytest

    # Run specific test file
    pytest tests/test_cola.py

    # Run with coverage
    pytest --cov=xai_cola tests/

    # Run with verbose output
    pytest -v tests/

Documentation
=============

Building Docs
-------------

.. code-block:: bash

    cd docs
    make html

    # View docs
    open _build/html/index.html  # macOS
    # or
    start _build/html/index.html  # Windows

Writing Docs
------------

Documentation is in reStructuredText (.rst) format.

**Example:**

.. code-block:: rst

    =========
    My Title
    =========

    Section
    =======

    Subsection
    ----------

    Some text with **bold** and *italic*.

    .. code-block:: python

        # Python code example
        from xai_cola import COLA

    See :doc:`other_page` for more info.

Pull Request Process
====================

1. **Update Documentation**

   Ensure all new features are documented.

2. **Add Tests**

   Maintain or improve test coverage.

3. **Update CHANGELOG.md**

   Add entry under "Unreleased" section.

4. **Ensure CI Passes**

   All tests and checks must pass.

5. **Request Review**

   Tag maintainers for review.

6. **Address Feedback**

   Make requested changes promptly.

7. **Squash Commits (if requested)**

   .. code-block:: bash

       git rebase -i HEAD~n
       # Squash commits
       git push --force-with-lease

Code Review Guidelines
======================

As a Reviewer
-------------

- Be constructive and respectful
- Explain *why* changes are needed
- Suggest alternatives
- Approve promptly if everything looks good

As an Author
------------

- Be open to feedback
- Ask questions if unclear
- Make requested changes
- Thank reviewers for their time

Community Guidelines
====================

Be Respectful
-------------

- Use welcoming and inclusive language
- Respect differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community

Be Professional
---------------

- Keep discussions on-topic
- Avoid personal attacks
- Assume good intentions
- Be patient with newcomers

Release Process
===============

For Maintainers
---------------

1. **Update version in VERSION file**

2. **Update CHANGELOG.md**

   Move "Unreleased" changes to new version section.

3. **Create release branch**

   .. code-block:: bash

       git checkout -b release/v0.2.0

4. **Tag release**

   .. code-block:: bash

       git tag -a v0.2.0 -m "Release v0.2.0"
       git push origin v0.2.0

5. **Build and upload to PyPI**

   .. code-block:: bash

       python -m build
       twine upload dist/*

6. **Create GitHub release**

   Add release notes from CHANGELOG.

Getting Help
============

If you need help:

- Check the :doc:`faq`
- Read existing documentation
- Search `GitHub Issues <https://github.com/understanding-ml/COLA/issues>`_
- Ask in a new issue
- Email: leiyo@dtu.dk, s232291@dtu.dk

Recognition
===========

Contributors will be:

- Added to the CONTRIBUTORS file
- Mentioned in release notes
- Acknowledged in the documentation

Thank you for contributing to COLA! 🎉
