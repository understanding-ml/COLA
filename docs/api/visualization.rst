==================
Visualization API
==================

.. currentmodule:: xai_cola.ce_sparsifier.visualization

Highlighted DataFrames
======================

.. autofunction:: highlight_changes_final

.. autofunction:: highlight_changes_comparison

.. autofunction:: highlight_differences

Heatmaps
========

Direction Heatmap
-----------------

.. autofunction:: generate_direction_heatmap

.. autofunction:: heatmap_direction_changes

Binary Heatmap
--------------

.. autofunction:: generate_binary_heatmap

.. autofunction:: heatmap_binary_changes

Stacked Bar Chart
=================

.. autofunction:: generate_stacked_bar_chart

.. autofunction:: create_stacked_bar_chart

Diversity Analysis
==================

.. autofunction:: generate_diversity_for_all_instances

.. autofunction:: generate_diversity_dataframe

.. autofunction:: highlight_diversity_changes

.. autofunction:: find_minimal_feature_combinations

Examples
========

Direction Heatmap
-----------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from xai_cola.ce_sparsifier.visualization import generate_direction_heatmap

    fig = generate_direction_heatmap(
        factual_df=factual,
        cf_df=counterfactual,
        ace_df=refined_counterfactual,
        label_column='Risk',
        save_path='./results',
        save_mode='combined'
    )

Via COLA Instance
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from xai_cola import COLA

    cola = COLA(data=data, ml_model=ml_model)
    cola.set_policy(matcher="ot", attributor="pshap")
    cola.refine_counterfactuals(limited_actions=5)

    # Generate heatmap
    fig = cola.heatmap_direction(
        save_path='./results',
        save_mode='combined',
        show_axis_labels=True
    )

Binary Heatmap
--------------

.. code-block:: python

    from xai_cola.ce_sparsifier.visualization import generate_binary_heatmap

    fig = generate_binary_heatmap(
        factual_df=factual,
        cf_df=counterfactual,
        ace_df=refined_counterfactual,
        label_column='Risk',
        save_path='./results'
    )

Stacked Bar Chart
-----------------

.. code-block:: python

    from xai_cola.ce_sparsifier.visualization import generate_stacked_bar_chart

    fig = generate_stacked_bar_chart(
        factual_df=factual,
        cf_df=counterfactual,
        ace_df=refined_counterfactual,
        label_column='Risk',
        save_path='./results'
    )

Highlighted Comparison
----------------------

.. code-block:: python

    from xai_cola.ce_sparsifier.visualization import highlight_changes_comparison

    styled_ce, styled_ace = highlight_changes_comparison(
        factual_df=factual,
        cf1_df=counterfactual,
        cf2_df=refined_counterfactual,
        label_column='Risk'
    )

    # Display in Jupyter
    display(styled_ce)
    display(styled_ace)

    # Save to HTML
    styled_ce.to_html('original_cf.html')
    styled_ace.to_html('refined_ace.html')

Diversity Analysis
------------------

Find All Minimal Combinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from xai_cola.ce_sparsifier.visualization import (
        generate_diversity_for_all_instances
    )

    diversity_results = generate_diversity_for_all_instances(
        factual_df=factual,
        ml_model=ml_model,
        target_class=0,
        max_features=5
    )

    # Results: dict mapping instance_id to list of feature combinations
    for instance_id, combinations in diversity_results.items():
        print(f"Instance {instance_id}:")
        for combo in combinations:
            print(f"  - {combo}")

Highlighted Diversity DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from xai_cola.ce_sparsifier.visualization import highlight_diversity_changes

    # Get highlighted diversity DataFrame
    diversity_styled = highlight_diversity_changes(
        factual_df=factual,
        diversity_results=diversity_results,
        label_column='Risk'
    )

    display(diversity_styled)

Complete Visualization Suite
-----------------------------

Generate All Visualizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    from xai_cola import COLA

    # Setup and refine
    cola = COLA(data=data, ml_model=ml_model)
    cola.set_policy(matcher="ot", attributor="pshap", random_state=42)
    cola.refine_counterfactuals(limited_actions=5)

    # Create output directory
    os.makedirs('./results', exist_ok=True)

    # Generate all visualizations
    cola.heatmap_direction(save_path='./results', save_mode='both')
    cola.heatmap_binary(save_path='./results', save_mode='both')
    cola.stacked_bar_chart(save_path='./results')

    # Highlighted DataFrames
    factual_style, ce_style, ace_style = cola.highlight_changes_final()
    ce_style.to_html('./results/original_cf.html')
    ace_style.to_html('./results/refined_ace.html')

    print("✓ All visualizations saved to ./results/")

Customization
=============

Figure Size and DPI
-------------------

.. code-block:: python

    # High resolution for publications
    fig = cola.heatmap_direction(
        save_path='./results',
        figsize=(14, 10),
        dpi=300
    )

    # Large size for presentations
    fig = cola.stacked_bar_chart(
        save_path='./results',
        figsize=(18, 10),
        dpi=150
    )

Axis Labels
-----------

.. code-block:: python

    # Show feature and instance names
    fig = cola.heatmap_direction(
        save_path='./results',
        show_axis_labels=True
    )

    # Hide for cleaner look
    fig = cola.heatmap_direction(
        save_path='./results',
        show_axis_labels=False
    )

Save Modes
----------

.. code-block:: python

    # Combined: CE and ACE side by side
    cola.heatmap_direction(save_mode='combined')

    # Separate: Two individual heatmaps
    cola.heatmap_direction(save_mode='separate')

    # Both: Generate both combined and separate
    cola.heatmap_direction(save_mode='both')

Multiple Output Formats
-----------------------

.. code-block:: python

    # Generate figure
    fig = cola.heatmap_direction(save_path='./results')

    # Save in multiple formats
    fig.savefig('./results/heatmap.png', dpi=300, bbox_inches='tight')
    fig.savefig('./results/heatmap.pdf', bbox_inches='tight')
    fig.savefig('./results/heatmap.svg', bbox_inches='tight')

Color Schemes
-------------

For direction heatmaps, colors indicate:

- 🟦 **Blue** - Feature value increased
- 🟧 **Red/Orange** - Feature value decreased
- ⬜ **White** - No change

For binary heatmaps:

- ⬛ **Black** - Feature changed
- ⬜ **White** - No change

For highlighted DataFrames:

- 🟦 **Blue background** - Value increased
- 🟧 **Orange background** - Value decreased
- ⬜ **White background** - No change

See Also
========

- :doc:`../user_guide/visualization` - Detailed visualization guide
- :doc:`cola` - COLA main class
- :doc:`ce_generator` - Counterfactual generators
