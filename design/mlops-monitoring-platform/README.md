# MLOps Monitoring Platform UI Mockup

High-fidelity desktop mockup for a production MLOps monitoring platform.

## Open the mockup

Open `index.html` in a browser. The page contains six 1440px desktop frames:

1. Main MLOps Dashboard
2. Model Detail
3. Data Drift Monitoring
4. Model Performance
5. Alerts and Incidents
6. Model Registry

## Included UI system

- Light enterprise SaaS visual style.
- Reusable CSS tokens and components for cards, tables, status badges, alert cards, metric cards, buttons, charts, sidebar navigation, and topbar controls.
- Realistic ML operations data for `churn_prediction_v3`, `fraud_detection_v2`, `demand_forecast_v1`, `credit_risk_v4`, and related alerts, drift scores, registry entries, and quality metrics.

## Figma handoff note

This repository artifact is structured as a Figma-ready design reference, but it is not a live Figma file. Creating or editing a Figma project requires authenticated Figma access or a Figma MCP/API token in the agent environment.

## Generate editable Figma frames

The `figma-plugin/` folder contains a local Figma plugin generator.

To create an editable Figma board:

1. Open Figma.
2. Go to `Plugins -> Development -> Import plugin from manifest...`.
3. Select `design/mlops-monitoring-platform/figma-plugin/manifest.json`.
4. Run `ModelPulse MLOps Dashboard Generator`.

The plugin creates six editable 1440px frames with reusable visual primitives, realistic data, charts, tables, status badges, alert cards, sidebar navigation, and topbar controls.
