"""Visual-audit harness for mlframe charts.

Renders every chart type x both backends (matplotlib + plotly) to a
local directory for manual inspection. Not a pytest test (visual
inspection cannot be automated reliably) -- run manually before
shipping any change to ``mlframe.reporting.charts`` or to renderers.

Usage::

    python -m mlframe.tests.reporting.visual_audit.render_all_charts \
        --out D:/Temp/chart_audit
"""
