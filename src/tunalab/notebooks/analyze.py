import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import yaml
    import glob
    from pathlib import Path

    from tunalab.analysis import (
        find_dashboards,
        load_dashboard,
        save_dashboard,
        Run,
        normalize_metrics,
    )

    return (
        Run,
        find_dashboards,
        glob,
        load_dashboard,
        mo,
        normalize_metrics,
        pd,
        px,
        save_dashboard,
        yaml,
        Path,
    )


@app.cell
def _(find_dashboards, mo):
    dashboards = find_dashboards(".")
    options = {str(p): str(p) for p in dashboards}

    dashboard_select = mo.ui.dropdown(
        options=options,
        label="Select Dashboard",
    )

    mo.md(f"### Select Dashboard\n{dashboard_select}")
    return dashboard_select, dashboards, options


@app.cell
def _(dashboard_select, load_dashboard, mo, yaml):
    if dashboard_select.value:
        initial_config = load_dashboard(dashboard_select.value)
    else:
        initial_config = {
            "name": "New Dashboard",
            "defaults": {"x_axis_key": None, "x_axis_scale": 1.0, "smoothing": 0.0},
            "experiments": [],
            "metrics": [],
        }

    name_input = mo.ui.text(
        value=initial_config.get("name", ""), 
        label="Dashboard Name"
    )

    defaults = initial_config.get("defaults", {})
    x_axis_key_input = mo.ui.text(
        value=defaults.get("x_axis_key") or "", 
        label="X-Axis Key (empty for index)"
    )
    x_axis_scale_input = mo.ui.number(
        value=float(defaults.get("x_axis_scale", 1.0)), 
        label="X-Axis Scale"
    )
    smoothing_input = mo.ui.number(
        value=float(defaults.get("smoothing", 0.0)),
        step=0.1,
        start=0.0,
        label="Smoothing",
    )

    experiments_list = initial_config.get("experiments", [])
    experiments_text = "\n".join(experiments_list)
    experiments_input = mo.ui.text_area(
        value=experiments_text,
        label="Experiment Globs (one per line)",
        full_width=True,
    )

    metrics_list = initial_config.get("metrics", [])
    metrics_yaml = yaml.dump(metrics_list, default_flow_style=False, sort_keys=False)
    metrics_input = mo.ui.code_editor(
        value=metrics_yaml, language="yaml", label="Metrics Configuration"
    )

    save_btn = mo.ui.button(label="Save Configuration")

    mo.md(
        f"""
    ## Configuration
    {name_input}
    
    ### Defaults
    <div style="display: flex; gap: 1rem;">
        {x_axis_key_input}
        {x_axis_scale_input}
        {smoothing_input}
    </div>
    
    ### Experiments
    {experiments_input}
    
    ### Metrics
    {metrics_input}
    
    <br/>
    {save_btn}
    """
    )

    return (
        defaults,
        experiments_input,
        experiments_list,
        experiments_text,
        initial_config,
        metrics_input,
        metrics_list,
        metrics_yaml,
        name_input,
        save_btn,
        smoothing_input,
        x_axis_key_input,
        x_axis_scale_input,
    )


@app.cell
def _(
    dashboard_select,
    experiments_input,
    metrics_input,
    mo,
    name_input,
    save_btn,
    save_dashboard,
    smoothing_input,
    x_axis_key_input,
    x_axis_scale_input,
    yaml,
):
    try:
        current_metrics = yaml.safe_load(metrics_input.value) or []
    except yaml.YAMLError:
        current_metrics = []

    current_experiments = [
        line.strip() for line in experiments_input.value.split("\n") if line.strip()
    ]

    x_key_val = x_axis_key_input.value.strip()
    current_defaults = {
        "x_axis_key": x_key_val if x_key_val else None,
        "x_axis_scale": x_axis_scale_input.value,
        "smoothing": smoothing_input.value,
    }

    current_config = {
        "name": name_input.value,
        "defaults": current_defaults,
        "experiments": current_experiments,
        "metrics": current_metrics,
    }

    if save_btn.value and dashboard_select.value:
        try:
            save_dashboard(dashboard_select.value, current_config)
            mo.output.replace(
                mo.md(f"✅ **Configuration saved to `{dashboard_select.value}`**")
            )
        except Exception as e:
            mo.output.replace(mo.md(f"❌ **Error saving configuration:** {e}"))

    return (
        current_config,
        current_defaults,
        current_experiments,
        current_metrics,
        x_key_val,
    )


@app.cell
def _(
    Run,
    current_config,
    current_experiments,
    glob,
    mo,
    normalize_metrics,
):
    files = []
    for pattern in current_experiments:
        files.extend(glob.glob(pattern, recursive=True))

    files = sorted(list(set(files)))

    runs = []
    errors = []

    if not files:
        mo.md("⚠️ No experiment files found matching the patterns.")
        normalized_data = {}
    else:
        with mo.status.spinner(f"Loading {len(files)} runs..."):
            for f in files:
                try:
                    runs.append(Run.from_path(f))
                except Exception as e:
                    errors.append(f"{f}: {e}")

        normalized_data = normalize_metrics(runs, current_config["metrics"])

    if errors:
        error_list = "\n".join([f"- {e}" for e in errors])
        mo.md(f"**Errors loading files:**\n{error_list}")

    return errors, files, normalized_data, runs


@app.cell
def _(current_config, mo, normalized_data, pd, px):
    tabs = None
    scalar_df = None
    all_curves_long = None

    if normalized_data:
        tabs_content = {}

        # --- Tab 1: Scalars ---
        scalar_records = []
        metric_names = [m["name"] for m in current_config["metrics"]]

        for run_id, df in normalized_data.items():
            for col in df.columns:
                if col in metric_names:
                    valid = df[col].dropna()
                    if not valid.empty:
                        scalar_records.append(
                            {
                                "Run": str(run_id),
                                "Metric": col,
                                "Last": valid.iloc[-1],
                                "Best (Max)": valid.max(),
                                "Best (Min)": valid.min(),
                            }
                        )

        if scalar_records:
            scalar_df = pd.DataFrame(scalar_records)
            fig_scalars = px.bar(
                scalar_df,
                x="Run",
                y="Last",
                color="Metric",
                barmode="group",
                title="Last Value per Metric",
            )
            tabs_content["Scalars"] = fig_scalars
        else:
            tabs_content["Scalars"] = mo.md("No scalar data found.")

        # --- Tab 2: Curves ---
        curve_dfs = []
        _defaults = current_config["defaults"]
        x_key = _defaults["x_axis_key"]
        x_scale = _defaults["x_axis_scale"]

        for run_id, df in normalized_data.items():
            # Determine X
            if x_key and x_key in df.columns:
                x_values = df[x_key]
            else:
                x_values = df.index

            # Scale X
            try:
                # Handle index vs series
                if hasattr(x_values, "to_series"):
                    x_values = x_values.to_series()
                x_values = x_values.astype(float) * x_scale
            except (ValueError, TypeError):
                pass  # Keep original if conversion fails

            # Get metric columns
            plot_cols = [c for c in df.columns if c in metric_names]

            if plot_cols:
                subset = df[plot_cols].copy()
                subset["_x"] = x_values
                subset["Run"] = str(run_id)
                curve_dfs.append(subset)

        if curve_dfs:
            all_curves = pd.concat(curve_dfs)
            all_curves_long = all_curves.melt(
                id_vars=["_x", "Run"], var_name="Metric", value_name="Value"
            )

            fig_curves = px.line(
                all_curves_long,
                x="_x",
                y="Value",
                color="Run",
                line_dash="Metric",
                title=f"Curves (X: {x_key or 'Index'} * {x_scale})",
            )
            tabs_content["Curves"] = fig_curves
        else:
            tabs_content["Curves"] = mo.md("No curve data found.")

        tabs = mo.ui.tabs(tabs_content)
    
    return all_curves_long, scalar_df, tabs


@app.cell
def _(tabs):
    tabs
    return


if __name__ == "__main__":
    app.run()
