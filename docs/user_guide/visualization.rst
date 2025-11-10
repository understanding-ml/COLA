==============
Visualization
==============

Overview
========

COLA provides rich visualization tools to help you understand and communicate counterfactual refinement results. These visualizations show:

- Which features were changed
- How features were changed (increase/decrease)
- How many actions are required
- Diversity of counterfactual options

Available Visualizations
========================

COLA offers five main visualization types:

1. **Highlighted DataFrames** - Side-by-side comparison with color highlighting
2. **Direction Heatmap** - Show which features increase/decrease
3. **Binary Heatmap** - Show which features changed
4. **Stacked Bar Chart** - Compare action counts before/after refinement
5. **Diversity Analysis** - Explore alternative minimal feature combinations

Quick Start
===========

.. code-block:: python

    from xai_cola import COLA
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model

    # Setup and refine
    sparsifier = COLA(data=data, ml_model=ml_model)
    sparsifier.set_policy(matcher="ot", attributor="pshap")
    refined = sparsifier.refine_counterfactuals(limited_actions=5)

    # Visualize!
    sparsifier.heatmap_direction(save_path='./results')
    sparsifier.stacked_bar_chart(save_path='./results')

1. Highlighted DataFrames
=========================

Show factual and counterfactual data side-by-side with color-coded changes.

Highlight Final Comparison
---------------------------

Compare factual, original CF, and refined ACE:

.. code-block:: python

    # Get highlighted DataFrames
    factual_style, ce_style, ace_style = sparsifier.highlight_changes_final()

    # Display in Jupyter
    display(factual_style)  # Original data
    display(ce_style)       # Original counterfactual (more changes)
    display(ace_style)      # Refined counterfactual (fewer changes)

    # Save to HTML
    ce_style.to_html('original_cf.html')
    ace_style.to_html('refined_ace.html')

.. image:: ../images/highlight_changed_positions.png
   :width: 600
   :alt: Highlighted DataFrame comparison

**Color coding:**

- 🟦 **Blue** - Feature increased
- 🟧 **Orange** - Feature decreased
- ⬜ **White** - No change

**When to use:**

- Presenting to stakeholders
- Detailed instance-by-instance analysis
- Generating reports

Highlight Changes Comparison
-----------------------------

Compare two specific counterfactuals:

.. code-block:: python

    from xai_cola.ce_sparsifier.visualization import highlight_changes_comparison

    # Compare any two DataFrames
    factual_df, ce_df, ace_df = sparsifier.get_all_results(limited_actions=5)

    styled_ce, styled_ace = highlight_changes_comparison(
        factual_df=factual_df,
        cf1_df=ce_df,
        cf2_df=ace_df,
        label_column='Risk'
    )

    display(styled_ce)
    display(styled_ace)

2. Direction Heatmap
====================

Visualize the direction of feature changes (increase vs decrease) across instances.

Basic Usage
-----------

.. code-block:: python

    # Generate direction heatmap
    fig = sparsifier.heatmap_direction(
        save_path='./results',
        save_mode='combined',      # 'combined', 'separate', or 'both'
        show_axis_labels=True,     # Show feature and instance names
        figsize=(12, 8)
    )

.. image:: ../images/combined_direction_heatmap.png
   :width: 600
   :alt: Direction heatmap

**Color coding:**

- 🟦 **Blue** - Feature increased
- 🟧 **Red/Orange** - Feature decreased
- ⬜ **White** - No change

Save Modes
----------

**Combined mode** - CE and ACE side by side:

.. code-block:: python

    fig = sparsifier.heatmap_direction(
        save_path='./results',
        save_mode='combined'  # Default
    )

**Separate mode** - Two separate heatmaps:

.. code-block:: python

    fig = sparsifier.heatmap_direction(
        save_path='./results',
        save_mode='separate'
    )
    # Creates: heatmap_direction_counterfactual.png
    #          heatmap_direction_counterfactual_with_limited_actions.png

**Both modes**:

.. code-block:: python

    fig = sparsifier.heatmap_direction(
        save_path='./results',
        save_mode='both'
    )

Customization
-------------

.. code-block:: python

    fig = sparsifier.heatmap_direction(
        save_path='./results',
        save_mode='combined',
        show_axis_labels=False,  # Hide labels for cleaner look
        figsize=(16, 10),        # Larger figure
        dpi=300                  # Higher resolution
    )

**When to use:**

- Understanding feature change patterns
- Identifying which features typically increase/decrease
- Comparing CE vs ACE visually

3. Binary Heatmap
=================

Show which features changed (binary: changed or not changed).

Basic Usage
-----------

.. code-block:: python

    # Generate binary heatmap
    fig = sparsifier.heatmap_binary(
        save_path='./results',
        save_mode='combined'
    )

**Color coding:**

- ⬛ **Black** - Feature changed
- ⬜ **White** - No change

**When to use:**

- Focusing on which features changed, not how
- Counting total feature changes
- Simple visual comparison

4. Stacked Bar Chart
====================

Compare the number of feature changes before and after refinement.

Basic Usage
-----------

.. code-block:: python

    # Generate stacked bar chart
    fig = sparsifier.stacked_bar_chart(
        save_path='./results',
        figsize=(14, 6)
    )

.. image:: ../images/stacked_bar_chart.png
   :width: 600
   :alt: Stacked bar chart

**What it shows:**

- X-axis: Each factual instance
- Y-axis: Number of feature changes
- Colors: Different features that changed
- Two bars per instance: Original CF vs Refined ACE

**Interpretation:**

.. code-block:: text

    Instance 1: ████████ (8 changes)  →  ███ (3 changes)  ✓ Reduced by 5!
    Instance 2: ██████ (6 changes)    →  ██ (2 changes)   ✓ Reduced by 4!

**When to use:**

- Demonstrating COLA's effectiveness
- Showing reduction in required actions
- Presentations and papers

Customization
-------------

.. code-block:: python

    fig = sparsifier.stacked_bar_chart(
        save_path='./results',
        figsize=(16, 8),
        dpi=300,
        show_legend=True
    )

5. Diversity Analysis
=====================

Explore alternative minimal feature combinations for achieving the same outcome.

Basic Usage
-----------

.. code-block:: python

    # Find all minimal feature combinations
    diversity_results = sparsifier.diversity_analysis(limited_actions=5)

    # Get highlighted DataFrame
    diversity_df = sparsifier.highlight_diversity_changes()
    display(diversity_df)

**What it shows:**

For each instance, shows multiple alternative ways to achieve the desired outcome using the same number of features.

Example Output
--------------

.. code-block:: text

    Instance 0:
      Option 1: Change [Income, Duration]
      Option 2: Change [Income, LoanAmount]
      Option 3: Change [Age, Duration]

    Instance 1:
      Option 1: Change [Income, Job]
      Option 2: Change [Duration, Housing]

**When to use:**

- Providing users with choices
- Understanding feature importance
- Exploring alternative action plans

Advanced Usage
--------------

.. code-block:: python

    from xai_cola.ce_sparsifier.visualization import (
        generate_diversity_for_all_instances
    )

    # Generate diversity analysis for all instances
    all_diversity = generate_diversity_for_all_instances(
        factual_df=factual_df,
        ml_model=ml_model,
        target_class=0,
        max_features=5
    )

    # Analyze results
    for instance_id, combinations in all_diversity.items():
        print(f"\nInstance {instance_id}:")
        print(f"  Found {len(combinations)} minimal combinations")
        for i, combo in enumerate(combinations, 1):
            print(f"  Option {i}: {combo}")

Complete Visualization Workflow
================================

End-to-End Example
------------------

.. code-block:: python

    import os
    from xai_cola import COLA
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model
    from xai_cola.ce_generator import DiCE

    # 1. Setup
    data = COLAData(factual_data=df, label_column='Risk')
    ml_model = Model(model=trained_model, backend="sklearn")

    # 2. Generate counterfactuals
    explainer = DiCE(ml_model=ml_model)
    _, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2
    )
    data.add_counterfactuals(cf, with_target_column=True)

    # 3. Refine
    sparsifier = COLA(data=data, ml_model=ml_model)
    sparsifier.set_policy(matcher="ot", attributor="pshap", random_state=42)
    min_actions = sparsifier.query_minimum_actions()
    refined = sparsifier.refine_counterfactuals(limited_actions=min_actions)

    # 4. Create results directory
    os.makedirs('./results', exist_ok=True)

    # 5. Generate all visualizations
    print("Generating visualizations...")

    # Heatmaps
    sparsifier.heatmap_direction(
        save_path='./results',
        save_mode='both',
        show_axis_labels=True
    )
    print("✓ Direction heatmap saved")

    sparsifier.heatmap_binary(
        save_path='./results',
        save_mode='both'
    )
    print("✓ Binary heatmap saved")

    # Stacked bar chart
    sparsifier.stacked_bar_chart(save_path='./results')
    print("✓ Stacked bar chart saved")

    # Highlighted DataFrames
    factual_style, ce_style, ace_style = sparsifier.highlight_changes_final()
    ce_style.to_html('./results/original_counterfactual.html')
    ace_style.to_html('./results/refined_counterfactual.html')
    print("✓ Highlighted DataFrames saved")

    # Diversity analysis
    diversity_df = sparsifier.highlight_diversity_changes()
    diversity_df.to_html('./results/diversity_analysis.html')
    print("✓ Diversity analysis saved")

    print("\n✓ All visualizations complete!")
    print("Check ./results/ directory for outputs")

For Publications
----------------

High-quality figures for papers:

.. code-block:: python

    # High resolution, no axis labels for clean look
    sparsifier.heatmap_direction(
        save_path='./paper_figures',
        save_mode='combined',
        show_axis_labels=False,
        figsize=(10, 6),
        dpi=300
    )

    sparsifier.stacked_bar_chart(
        save_path='./paper_figures',
        figsize=(12, 5),
        dpi=300
    )

For Presentations
-----------------

Larger, easy-to-read figures:

.. code-block:: python

    # Bigger font, higher contrast
    sparsifier.heatmap_direction(
        save_path='./presentation',
        save_mode='combined',
        show_axis_labels=True,
        figsize=(16, 10),
        dpi=150
    )

    sparsifier.stacked_bar_chart(
        save_path='./presentation',
        figsize=(16, 8),
        dpi=150
    )

Common Issues
=============

Issue 1: Figures Too Small
---------------------------

**Problem:** Text is unreadable in saved figures.

**Solution:** Increase figsize and DPI:

.. code-block:: python

    # ❌ Too small
    fig = sparsifier.heatmap_direction(figsize=(6, 4))

    # ✅ Better
    fig = sparsifier.heatmap_direction(figsize=(16, 10), dpi=300)

Issue 2: Colors Not Showing
----------------------------

**Problem:** Heatmap appears all white.

**Cause:** No features changed.

**Solution:** Check if refinement actually changed anything:

.. code-block:: python

    factual_df, ce_df, ace_df = sparsifier.get_all_results(limited_actions=5)

    # Count changes
    changes = (factual_df != ace_df).sum().sum()
    print(f"Total feature changes: {changes}")

    if changes == 0:
        print("No changes - increase limited_actions")

Issue 3: HTML Not Displaying
-----------------------------

**Problem:** HTML files don't show styling.

**Solution:** Use ``display()`` in Jupyter or open HTML directly in browser:

.. code-block:: python

    # In Jupyter
    from IPython.display import display
    display(styled_df)

    # Or save and open in browser
    styled_df.to_html('results.html')
    # Then open results.html in Chrome/Firefox

Issue 4: Memory Error with Large Datasets
------------------------------------------

**Problem:** OutOfMemoryError when generating visualizations.

**Solution:** Visualize a subset:

.. code-block:: python

    # Get results
    factual_df, ce_df, ace_df = sparsifier.get_all_results(limited_actions=5)

    # Visualize first 50 instances
    from xai_cola.ce_sparsifier.visualization import generate_direction_heatmap

    fig = generate_direction_heatmap(
        factual_df=factual_df.head(50),
        cf_df=ce_df.head(50),
        ace_df=ace_df.head(50),
        label_column='Risk',
        save_path='./results'
    )

Best Practices
==============

✅ **DO:**

1. **Generate multiple visualization types** for comprehensive understanding

   .. code-block:: python

       # Show different aspects
       sparsifier.heatmap_direction(save_path='./results')  # How
       sparsifier.stacked_bar_chart(save_path='./results')  # How many
       sparsifier.highlight_changes_final()                 # Details

2. **Use appropriate figure sizes** for your medium

   .. code-block:: python

       # Paper: smaller, high DPI
       figsize=(10, 6), dpi=300

       # Presentation: larger, medium DPI
       figsize=(16, 10), dpi=150

       # Web: medium, low DPI
       figsize=(12, 8), dpi=96

3. **Save in multiple formats** for flexibility

   .. code-block:: python

       fig = sparsifier.heatmap_direction(save_path='./results')
       fig.savefig('./results/heatmap.png', dpi=300, bbox_inches='tight')
       fig.savefig('./results/heatmap.pdf', bbox_inches='tight')  # Vector
       fig.savefig('./results/heatmap.svg', bbox_inches='tight')  # Vector

4. **Create a results directory** before saving

   .. code-block:: python

       import os
       os.makedirs('./results', exist_ok=True)

❌ **DON'T:**

1. **Don't forget to refine before visualizing**

   .. code-block:: python

       # ❌ Error - no refinement yet
       sparsifier = COLA(data=data, ml_model=ml_model)
       sparsifier.heatmap_direction()

       # ✅ Correct
       sparsifier.set_policy(matcher="ot", attributor="pshap")
       sparsifier.refine_counterfactuals(limited_actions=5)
       sparsifier.heatmap_direction()

2. **Don't use show_axis_labels=True with many features** - too cluttered

3. **Don't generate huge figures** - stick to reasonable sizes

API Reference
=============

For complete parameter details, see:

- :func:`~xai_cola.ce_sparsifier.visualization.generate_direction_heatmap`
- :func:`~xai_cola.ce_sparsifier.visualization.generate_binary_heatmap`
- :func:`~xai_cola.ce_sparsifier.visualization.generate_stacked_bar_chart`
- :func:`~xai_cola.ce_sparsifier.visualization.highlight_changes_final`
- :func:`~xai_cola.ce_sparsifier.visualization.generate_diversity_for_all_instances`

Next Steps
==========

- Review complete examples in :doc:`../tutorials/01_basic_tutorial`
- See :doc:`matching_policies` for refinement strategies
- Check :doc:`explainers` for CF generation methods
