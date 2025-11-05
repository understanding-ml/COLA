COLA Documentation
==================

.. image:: https://img.shields.io/badge/arXiv-2410.05419-B31B1B.svg
   :target: https://arxiv.org/pdf/2410.05419
   :alt: arXiv

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT License

.. image:: https://img.shields.io/pypi/v/xai-cola.svg
   :target: https://pypi.org/project/xai-cola/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.7+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.7+

**COLA** (COunterfactual explanations with Limited Actions) is a Python framework that refines counterfactual explanations by generating action-limited plans that require significantly fewer feature changes while maintaining similar or equal outcomes.

.. image:: images/problem.png
   :width: 700
   :align: center
   :alt: COLA Framework

Key Features
------------

- **Fewer Actions**: Reduces the number of feature changes needed by 30-70% compared to raw counterfactuals
- **Model Agnostic**: Works with any ML model (sklearn, PyTorch, TensorFlow)
- **CE Method Agnostic**: Compatible with various counterfactual explainers (DiCE, DisCount, Alibi, etc.)
- **Theoretically Grounded**: Based on joint-distribution-informed Shapley values (PSHAP)
- **Easy to Use**: Simple API with sensible defaults
- **Rich Visualizations**: Heatmaps, highlighted DataFrames, diversity analysis

Quick Start
-----------

Install COLA:

.. code-block:: bash

   pip install xai-cola

Basic usage:

.. code-block:: python

   from xai_cola import COLA
   from xai_cola.data import COLAData
   from xai_cola.models import Model
   from xai_cola.ce_generator import DiCE

   # 1. Prepare data
   data = COLAData(factual_data=df, label_column='target')

   # 2. Wrap model
   ml_model = Model(model=your_model, backend="sklearn")

   # 3. Generate counterfactuals
   explainer = DiCE(ml_model=ml_model)
   factual, counterfactual = explainer.generate_counterfactuals(
       data=data, factual_class=1, total_cfs=1
   )

   # 4. Refine with COLA
   data.add_counterfactuals(counterfactual, with_target_column=True)
   refiner = COLA(data=data, ml_model=ml_model)
   refiner.set_policy(matcher="ect", attributor="pshap")

   factual, ce, ace = refiner.get_all_results(limited_actions=5)

   # 5. Visualize
   refiner.heatmap_binary(save_path='./results')

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/data_interface
   user_guide/models
   user_guide/explainers
   user_guide/matching_policies
   user_guide/visualization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/cola
   api/data
   api/models
   api/ce_generator
   api/policies
   api/visualization

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   faq
   contributing
   changelog
   architecture

Citation
--------

If you use COLA in your research, please cite:

.. code-block:: bibtex

   @article{you2024refining,
     title={Refining Counterfactual Explanations With Joint-Distribution-Informed Shapley Towards Actionable Minimality},
     author={You, Lei and Bian, Yijun and Cao, Lele},
     journal={arXiv preprint arXiv:2410.05419},
     year={2024}
   }

Contact
-------

- Lin Zhu (s232291@dtu.dk)
- Lei You (leiyo@dtu.dk)

License
-------

COLA is licensed under the MIT License. See the `LICENSE <https://github.com/your-repo/COLA/blob/main/LICENSE>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
