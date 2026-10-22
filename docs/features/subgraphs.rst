Subgraphs
=========

.. image:: ../_static/subgraphs.png
   :alt: An If operator with entries for its nested graphs and graph navigation breadcrumbs.
   :width: 480px

Explore graph attributes of operators such as **If**, **Loop** and **Scan**.
Click a subgraph entry inside the operator's rectangle, or **Open** in the
inspector. Entries support Tab followed by Enter or Space; dragging the
operator's header still moves it. Each graph in a graph-list attribute has its
own entry. Breadcrumbs return to enclosing graphs, including the main graph.

Nested graphs retain enclosing opsets and captured tensor/initializer
information. Captured values appear as input ports; captured initializers stay
in the parameter list. The inspector lists implicit **Captured inputs**.
**Refresh shapes** infers the enclosing model and updates the displayed subgraph,
including intermediate ``value_info`` retained in its native graph.
