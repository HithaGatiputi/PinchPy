"""Interactive Streamlit Web Interface for PinchPy (Phase 12)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Ensure src directory is in sys.path
_src_dir = str(Path(__file__).resolve().parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pinchpy.models.stream import Stream
from pinchpy.analysis.pinch_point import utility_targets
from pinchpy.analysis.benchmarks import BENCHMARK_CASES
from pinchpy.synthesis.network_design import design_network
from pinchpy.synthesis.network_evaluation import validate_network_pinch_rules
from pinchpy.visualization.composite_curve import plot_composite_curves
from pinchpy.visualization.network_plot import plot_network
from pinchpy.economics.cost import ExchangerCostModel, UtilityCostModel, cost_analysis
from pinchpy.economics.emissions import EmissionsModel, emissions_analysis, FUEL_EMISSION_FACTORS
from pinchpy.retrofit.diagnosis import diagnose_network
from pinchpy.retrofit.recommendations import plan_retrofit
from pinchpy.reporting.report import generate_report, export_report_html, export_report_markdown, export_report_json
from pinchpy.utils.io import load_streams_from_csv, save_streams_to_csv


def run_app():
    st.set_page_config(
        page_title="PinchPy — HEN Design & Pinch Analysis",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔥 PinchPy: Industrial Pinch Analysis & HEN Design")
    st.caption("A free, cost-aware, ML-accelerated Heat Exchanger Network synthesis and retrofit tool for small & medium industries.")

    # =========================================================================
    # SIDEBAR: Stream Input & Settings
    # =========================================================================
    st.sidebar.header("⚙️ Configuration")

    input_mode = st.sidebar.radio(
        "Stream Input Source",
        ["Preset Literature Benchmarks", "Manual Stream Editor", "Upload CSV"],
    )

    streams: list[Stream] = []

    if input_mode == "Preset Literature Benchmarks":
        benchmark_key = st.sidebar.selectbox(
            "Select Benchmark Case",
            list(BENCHMARK_CASES.keys()),
            format_func=lambda k: BENCHMARK_CASES[k]["title"],
        )
        bench = BENCHMARK_CASES[benchmark_key]
        streams = bench["streams"]
        st.sidebar.info(bench["citation"])

    elif input_mode == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload Streams CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                streams = load_streams_from_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
                st.sidebar.success(f"Loaded {len(streams)} streams successfully!")
            except Exception as err:
                st.sidebar.error(f"Error reading CSV: {err}")
        else:
            st.sidebar.info("Upload a CSV file with columns: id, type, supply_temperature, target_temperature, cp")
            # Default fallback
            streams = BENCHMARK_CASES["kemp_2007_4stream"]["streams"]

    else:  # Manual Stream Editor
        st.sidebar.markdown("**Default Streams:** 4-stream template loaded below.")
        default_data = [
            {"id": "H1", "type": "hot", "supply": 170.0, "target": 60.0, "cp": 3.0},
            {"id": "H2", "type": "hot", "supply": 150.0, "target": 30.0, "cp": 1.5},
            {"id": "C1", "type": "cold", "supply": 20.0, "target": 135.0, "cp": 2.0},
            {"id": "C2", "type": "cold", "supply": 80.0, "target": 140.0, "cp": 4.0},
        ]
        df_editor = st.sidebar.data_editor(
            pd.DataFrame(default_data),
            num_rows="dynamic",
            use_container_width=True,
        )
        streams = []
        for _, row in df_editor.iterrows():
            try:
                streams.append(
                    Stream(
                        str(row["id"]),
                        str(row["type"]).lower(),
                        float(row["supply"]),
                        float(row["target"]),
                        float(row["cp"]),
                    )
                )
            except Exception:
                pass

    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Process Parameters")
    delta_t_min = st.sidebar.slider("Minimum Approach Temp (ΔTmin, °C)", min_value=1.0, max_value=40.0, value=10.0, step=0.5)
    overall_u = st.sidebar.number_input("Overall Heat Transfer Coeff U (kW/m²·°C)", min_value=0.01, max_value=5.0, value=0.5, step=0.05)

    with st.sidebar.expander("💰 Economic Settings"):
        st.caption("Capital cost correlation: a + b * Area^c")
        cost_a = st.number_input("Base Cost 'a' ($)", value=10000.0, step=1000.0)
        cost_b = st.number_input("Area Coeff 'b' ($/m²^c)", value=800.0, step=100.0)
        cost_c = st.number_input("Area Exponent 'c'", value=0.8, step=0.05)
        mat_factor = st.selectbox("Material Factor (Fm)", [1.0, 1.7, 2.2, 3.0], format_func=lambda f: f"Fm = {f} ({'Carbon Steel' if f==1.0 else 'SS304' if f==1.7 else 'SS316' if f==2.2 else 'Titanium'})")
        hu_cost = st.number_input("Hot Utility Cost ($/kWh)", value=0.03, step=0.005)
        cu_cost = st.number_input("Cold Utility Cost ($/kWh)", value=0.005, step=0.001)
        hours = st.number_input("Annual Operating Hours (h/yr)", value=8000.0, step=500.0)
        discount_rate = st.number_input("Annual Discount Rate", value=0.10, step=0.01)
        plant_life = st.number_input("Plant Lifetime (years)", value=10, step=1)

    with st.sidebar.expander("🌱 Decarbonization & Emissions"):
        fuel_choice = st.selectbox("Hot Utility Fuel", list(FUEL_EMISSION_FACTORS.keys()), index=0)
        boiler_eff = st.slider("Boiler/Heater Thermal Efficiency", min_value=0.5, max_value=1.0, value=0.85, step=0.05)
        carbon_tax = st.number_input("Carbon Tax / Credit Price ($/t CO2)", value=50.0, step=10.0)

    # Models
    c_model = ExchangerCostModel(a=cost_a, b=cost_b, c=cost_c, material_factor=mat_factor)
    u_model = UtilityCostModel(
        hot_utility_unit_cost=hu_cost,
        cold_utility_unit_cost=cu_cost,
        annual_operating_hours=hours,
        discount_rate=discount_rate,
        plant_lifetime=int(plant_life),
    )
    e_model = EmissionsModel(
        fuel_type=fuel_choice,
        hot_utility_efficiency=boiler_eff,
        carbon_price=carbon_tax,
        annual_operating_hours=hours,
    )

    if not streams:
        st.warning("No streams available. Please add streams or select a benchmark case.")
        return

    # =========================================================================
    # CORE CALCULATIONS
    # =========================================================================
    targets = utility_targets(streams, delta_t_min)
    network = design_network(streams, delta_t_min, overall_heat_transfer_coefficient=overall_u)
    validation_report = validate_network_pinch_rules(network, streams, delta_t_min)
    cost_rep = cost_analysis(network=network, streams=streams, exchanger_cost_model=c_model, utility_cost_model=u_model)
    emiss_rep = emissions_analysis(network=network, streams=streams, emissions_model=e_model)
    full_report = generate_report(
        streams=streams,
        delta_t_min=delta_t_min,
        overall_heat_transfer_coefficient=overall_u,
        network=network,
        exchanger_cost_model=c_model,
        utility_cost_model=u_model,
        emissions_model=e_model,
    )

    # =========================================================================
    # TABS INTERFACE
    # =========================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Pinch Analysis & Curves",
        "🌐 HEN Design & Grid",
        "💵 Economics & CO2",
        "🔧 Retrofit Analysis",
        "📥 Downloadable Reports",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Pinch Analysis & Curves
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("🎯 Theoretical Pinch Targets")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pinch Temp (Hot / Cold)", f"{targets['pinch_hot_temperature']} / {targets['pinch_cold_temperature']} °C")
        col2.metric("Min Hot Utility (QH,min)", f"{targets['minimum_hot_utility']:.1f} kW")
        col3.metric("Min Cold Utility (QC,min)", f"{targets['minimum_cold_utility']:.1f} kW")
        col4.metric("Max Heat Recovery", f"{targets.get('heat_recovery', 0.0):.1f} kW")

        st.markdown("### Process Stream Inventory")
        stream_df = pd.DataFrame([
            {
                "ID": s.id,
                "Type": s.type.upper(),
                "Supply Temp (°C)": s.supply_temperature,
                "Target Temp (°C)": s.target_temperature,
                "CP (kW/°C)": s.cp,
                "Duty (kW)": s.duty(),
            }
            for s in streams
        ])
        st.dataframe(stream_df, use_container_width=True)

        st.markdown("### 📈 Composite Curves")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_composite_curves(streams, delta_t_min, ax=ax)
        st.pyplot(fig)

    # -------------------------------------------------------------------------
    # TAB 2: HEN Design & Grid
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("🌐 Synthesized Heat Exchanger Network")
        v_col1, v_col2, v_col3 = st.columns(3)
        v_col1.metric("Energy Balance Conserved", "✅ Valid" if validation_report["energy_balance_conserved"] else "❌ Error")
        v_col2.metric("Zero Cross-Pinch Heat", "✅ Respected" if validation_report["no_cross_pinch_transfer"] else "❌ Violated")
        v_col3.metric("Approach Gaps >= ΔTmin", "✅ Respected" if validation_report["delta_t_min_respected"] else "❌ Tight")

        st.markdown("### Grid Diagram")
        fig_net, ax_net = plt.subplots(figsize=(10, 5))
        plot_network(streams, network, ax=ax_net)
        st.pyplot(fig_net)

        st.markdown("### Equipment Schedule")
        eq_df = pd.DataFrame([
            {
                "Unit": e.name,
                "Type": "Utility" if e.is_utility else "Process",
                "Hot Stream": e.hot_stream,
                "Cold Stream": e.cold_stream,
                "Duty (kW)": round(e.duty, 1),
                "Hot In (°C)": round(e.hot_inlet, 1) if e.hot_inlet is not None else "-",
                "Hot Out (°C)": round(e.hot_outlet, 1) if e.hot_outlet is not None else "-",
                "Cold In (°C)": round(e.cold_inlet, 1) if e.cold_inlet is not None else "-",
                "Cold Out (°C)": round(e.cold_outlet, 1) if e.cold_outlet is not None else "-",
                "Area (m²)": round(e.area, 2) if e.area is not None else "-",
                "Cost ($)": f"${c_model.exchanger_cost(e.area):,.0f}" if e.area is not None else "-",
            }
            for e in network.exchangers
        ])
        st.dataframe(eq_df, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: Economics & CO2
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("💵 Techno-Economic Feasibility")
        ec1, ec2, ec3, ec4 = st.columns(4)
        capex_val = cost_rep["installed_capital_cost_usd"]
        opex_val = cost_rep["annual_operating_cost_usd_per_yr"]
        sav_val = cost_rep["annual_operating_savings_usd_per_yr"]
        pb_val = cost_rep["simple_payback_years"]

        ec1.metric("Installed CAPEX", f"${capex_val:,.0f}" if capex_val else "N/A")
        ec2.metric("Annual Integrated OPEX", f"${opex_val:,.0f}/yr")
        ec3.metric("Annual Utility Savings", f"${sav_val:,.0f}/yr" if sav_val else "N/A")
        ec4.metric("Simple Payback", f"{pb_val:.2f} yrs" if pb_val else "N/A")

        st.markdown("---")
        st.subheader("🌱 Decarbonization & CO2 Savings")
        em1, em2, em3, em4 = st.columns(4)
        co2_act = emiss_rep["actual_annual_co2_tonnes"]
        co2_red = emiss_rep["annual_co2_reduction_tonnes"]
        pct_red = emiss_rep["co2_reduction_percentage"]
        tax_sav = emiss_rep["annual_carbon_cost_savings_usd"]

        em1.metric("Actual CO2 Emissions", f"{co2_act:,.1f} t/yr")
        em2.metric("CO2 Avoided", f"{co2_red:,.1f} t/yr" if co2_red else "N/A")
        em3.metric("Decarbonization Rate", f"{pct_red:.1f}%" if pct_red else "N/A")
        em4.metric("Carbon Tax Savings", f"${tax_sav:,.0f}/yr" if tax_sav else "N/A")

        with st.expander("📋 Review Modeling Assumptions"):
            st.json(cost_rep["assumptions"])

    # -------------------------------------------------------------------------
    # TAB 4: Retrofit Analysis
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("🔧 Industrial Retrofit Evaluation")
        st.write("Diagnose an existing plant configuration against pinch principles to uncover energy savings.")

        # Diagnosis of existing network
        retrofit_res = plan_retrofit(
            existing_network=network,
            streams=streams,
            delta_t_min=delta_t_min,
            overall_heat_transfer_coefficient=overall_u,
            exchanger_cost_model=c_model,
            utility_cost_model=u_model,
            emissions_model=e_model,
        )

        diag_res = diagnose_network(network, streams, delta_t_min, overall_heat_transfer_coefficient=overall_u)

        if diag_res["is_pinch_compliant"]:
            st.success("🎉 Current network is already pinch-compliant! No cross-pinch heat or misplaced utilities detected.")
        else:
            st.warning(f"Detected {len(diag_res['violations'])} pinch violation(s) causing excess utility consumption.")
            for v in diag_res["violations"]:
                st.error(f"**[{v.violation_type.upper()}] {v.exchanger_name}:** {v.recommendation}")

            st.markdown("### Actionable Retrofit Modifications")
            for m in retrofit_res.modifications:
                st.info(f"👉 **{m.unit_name}**: {m.description} | Est. Cost: ${m.estimated_cost_usd:,.0f}")

            rf1, rf2, rf3 = st.columns(3)
            rf1.metric("Retrofit Energy Savings", f"{retrofit_res.energy_savings_kw:.1f} kW ({retrofit_res.energy_savings_percent:.1f}%)")
            rf2.metric("Annual Cost Savings", f"${retrofit_res.annual_cost_savings_usd:,.0f}/yr")
            rf3.metric("Estimated Retrofit CAPEX", f"${retrofit_res.estimated_retrofit_capex_usd:,.0f}")

    # -------------------------------------------------------------------------
    # TAB 5: Downloadable Reports
    # -------------------------------------------------------------------------
    with tab5:
        st.subheader("📥 Download Analysis Reports")
        st.write("Export complete engineering audit reports, equipment schedules, and stream tables.")

        r_col1, r_col2, r_col3, r_col4 = st.columns(4)

        html_content = export_report_html(full_report)
        r_col1.download_button(
            label="📄 Download HTML Report",
            data=html_content,
            file_name="pinchpy_engineering_report.html",
            mime="text/html",
        )

        md_content = export_report_markdown(full_report)
        r_col2.download_button(
            label="📝 Download Markdown Audit",
            data=md_content,
            file_name="pinchpy_audit_report.md",
            mime="text/markdown",
        )

        json_content = export_report_json(full_report)
        r_col3.download_button(
            label="📊 Download JSON Data",
            data=json_content,
            file_name="pinchpy_analysis_data.json",
            mime="application/json",
        )

        csv_buf = io.StringIO()
        save_streams_to_csv(streams, csv_buf)
        r_col4.download_button(
            label="📋 Download Streams CSV",
            data=csv_buf.getvalue(),
            file_name="pinchpy_streams.csv",
            mime="text/csv",
        )

        st.markdown("### Report Preview")
        st.markdown(md_content)


if __name__ == "__main__":
    run_app()

