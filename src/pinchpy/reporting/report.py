"""Engineering reporting and export engine for PinchPy (Phase 12).

Generates comprehensive, multi-section reports covering:
1. Executive Summary & Key Performance Indicators (KPIs).
2. Process Stream Inventory.
3. Pinch Analysis & Minimum Utility Targets.
4. Heat Exchanger Network Equipment Schedule.
5. Capital and Operational Economics (CAPEX, OPEX, TAC, Payback, NPV, ROI).
6. Decarbonization and CO2 Emissions Reduction.
7. Retrofit Diagnosis and Recommendations (optional).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..analysis.pinch_point import utility_targets
from ..analysis.area_target import area_target
from ..synthesis.network_design import design_network
from ..synthesis.network_evaluation import evaluate_network
from ..economics.cost import ExchangerCostModel, UtilityCostModel, cost_analysis
from ..economics.emissions import EmissionsModel, emissions_analysis
from ..retrofit.diagnosis import diagnose_network
from ..retrofit.recommendations import plan_retrofit


def generate_report(
    streams: List[Any],
    delta_t_min: float,
    overall_heat_transfer_coefficient: Optional[float] = None,
    network=None,
    existing_network=None,
    exchanger_cost_model: Optional[ExchangerCostModel] = None,
    utility_cost_model: Optional[UtilityCostModel] = None,
    emissions_model: Optional[EmissionsModel] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive engineering analysis report for PinchPy.

    Parameters
    ----------
    streams : list[Stream]
        Process streams.
    delta_t_min : float
        Minimum approach temperature (°C).
    overall_heat_transfer_coefficient : float, optional
        Overall heat transfer coefficient U in kW/(m^2*°C).
    network : HeatExchangerNetwork, optional
        Synthesized network. If None, synthesized automatically.
    existing_network : HeatExchangerNetwork, optional
        Existing network for retrofit evaluation.
    exchanger_cost_model : ExchangerCostModel, optional
        Parameters for capital cost correlation.
    utility_cost_model : UtilityCostModel, optional
        Parameters for utility costs and financial evaluation.
    emissions_model : EmissionsModel, optional
        Parameters for CO2 emission estimation.

    Returns
    -------
    dict
        Structured report dictionary.
    """
    if delta_t_min <= 0:
        raise ValueError(f"delta_t_min must be positive, got {delta_t_min}")

    c_model = exchanger_cost_model or ExchangerCostModel()
    u_model = utility_cost_model or UtilityCostModel()
    e_model = emissions_model or EmissionsModel()

    # 1. Pinch Targets
    targets = utility_targets(streams, delta_t_min)

    # 2. Network Synthesis (if not provided)
    if network is None and len(streams) > 0:
        net = design_network(
            streams,
            delta_t_min,
            overall_heat_transfer_coefficient=overall_heat_transfer_coefficient,
        )
    else:
        net = network

    # 3. Network Evaluation & Economics
    cost_rep = cost_analysis(
        network=net,
        streams=streams,
        delta_t_min=delta_t_min,
        overall_heat_transfer_coefficient=overall_heat_transfer_coefficient,
        exchanger_cost_model=c_model,
        utility_cost_model=u_model,
    )

    # 4. Emissions Analysis
    emiss_rep = emissions_analysis(
        network=net,
        streams=streams,
        delta_t_min=delta_t_min,
        emissions_model=e_model,
    )

    # 5. Stream Table Summary
    stream_rows = [
        {
            "id": s.id,
            "type": s.type,
            "supply_temperature_c": s.supply_temperature,
            "target_temperature_c": s.target_temperature,
            "cp_kw_per_c": s.cp,
            "duty_kw": s.duty(),
        }
        for s in streams
    ]
    total_hot_duty = sum(s.duty() for s in streams if s.type == "hot")
    total_cold_duty = sum(s.duty() for s in streams if s.type == "cold")

    # 6. Equipment Schedule
    exchangers_list = []
    if net is not None:
        for e in net.exchangers:
            area = getattr(e, "area", None)
            unit_cost = c_model.exchanger_cost(area) if area is not None else None
            exchangers_list.append({
                "name": e.name,
                "is_utility": getattr(e, "is_utility", False),
                "hot_stream": e.hot_stream,
                "cold_stream": e.cold_stream,
                "duty_kw": e.duty,
                "hot_inlet_c": getattr(e, "hot_inlet", None),
                "hot_outlet_c": getattr(e, "hot_outlet", None),
                "cold_inlet_c": getattr(e, "cold_inlet", None),
                "cold_outlet_c": getattr(e, "cold_outlet", None),
                "lmtd_c": getattr(e, "lmtd", None),
                "area_m2": area,
                "installed_cost_usd": unit_cost,
            })

    # 7. Retrofit Evaluation (optional)
    retrofit_rep = None
    if existing_network is not None:
        retrofit_proj = plan_retrofit(
            existing_network=existing_network,
            streams=streams,
            delta_t_min=delta_t_min,
            overall_heat_transfer_coefficient=overall_heat_transfer_coefficient,
            exchanger_cost_model=c_model,
            utility_cost_model=u_model,
            emissions_model=e_model,
        )
        retrofit_rep = {
            "is_already_optimal": retrofit_proj.is_already_optimal,
            "energy_savings_kw": retrofit_proj.energy_savings_kw,
            "energy_savings_percent": retrofit_proj.energy_savings_percent,
            "annual_cost_savings_usd": retrofit_proj.annual_cost_savings_usd,
            "estimated_retrofit_capex_usd": retrofit_proj.estimated_retrofit_capex_usd,
            "simple_payback_years": retrofit_proj.simple_payback_years,
            "net_present_value_usd": retrofit_proj.net_present_value_usd,
            "annual_co2_reduction_tonnes": retrofit_proj.annual_co2_reduction_tonnes,
            "modifications": [
                {
                    "action": m.action,
                    "unit_name": m.unit_name,
                    "description": m.description,
                    "energy_impact_kw": m.target_energy_impact_kw,
                    "estimated_cost_usd": m.estimated_cost_usd,
                }
                for m in retrofit_proj.modifications
            ],
        }

    return {
        "metadata": {
            "tool": "PinchPy",
            "version": "0.1.0",
            "timestamp": datetime.now().isoformat(),
            "delta_t_min": delta_t_min,
            "overall_heat_transfer_coefficient": overall_heat_transfer_coefficient,
        },
        "streams": {
            "count": len(streams),
            "total_hot_duty_kw": total_hot_duty,
            "total_cold_duty_kw": total_cold_duty,
            "stream_list": stream_rows,
        },
        "pinch_targets": targets,
        "equipment_schedule": exchangers_list,
        "economics": cost_rep,
        "emissions": emiss_rep,
        "retrofit": retrofit_rep,
    }


def export_report_markdown(report: Dict[str, Any]) -> str:
    """Format report dictionary as a GitHub-flavored Markdown engineering audit."""
    meta = report["metadata"]
    st = report["streams"]
    tgt = report["pinch_targets"]
    econ = report["economics"]
    emiss = report["emissions"]
    eq = report["equipment_schedule"]

    lines = [
        f"# PinchPy Engineering Analysis Report",
        f"**Generated:** {meta['timestamp']} | **Delta T Min:** {meta['delta_t_min']} °C\n",
        "## 1. Executive Summary",
        f"- **Pinch Temperature (Hot / Cold):** {tgt['pinch_hot_temperature']} °C / {tgt['pinch_cold_temperature']} °C",
        f"- **Minimum Hot Utility (QH,min):** {tgt['minimum_hot_utility']:.2f} kW",
        f"- **Minimum Cold Utility (QC,min):** {tgt['minimum_cold_utility']:.2f} kW",
        f"- **Maximum Process Heat Recovery:** {tgt.get('heat_recovery', 0.0):.2f} kW",
        f"- **Installed Capital Cost (CAPEX):** ${econ['installed_capital_cost_usd']:,.2f}" if econ.get("installed_capital_cost_usd") else "- **Installed Capital Cost (CAPEX):** N/A",
        f"- **Annual Operating Savings:** ${econ['annual_operating_savings_usd_per_yr']:,.2f}/yr" if econ.get("annual_operating_savings_usd_per_yr") else "- **Annual Operating Savings:** N/A",
        f"- **Simple Payback Period:** {econ['simple_payback_years']:.2f} years" if econ.get("simple_payback_years") else "- **Simple Payback Period:** N/A",
        f"- **Annual CO2 Avoided:** {emiss['annual_co2_reduction_tonnes']:.2f} metric tonnes/yr ({emiss['co2_reduction_percentage']:.1f}% reduction)\n",
        "## 2. Process Stream Inventory",
        "| Stream ID | Type | Supply Temp (°C) | Target Temp (°C) | CP (kW/°C) | Duty (kW) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for s in st["stream_list"]:
        lines.append(f"| {s['id']} | {s['type']} | {s['supply_temperature_c']} | {s['target_temperature_c']} | {s['cp_kw_per_c']} | {s['duty_kw']:.2f} |")

    lines.extend([
        "",
        "## 3. Heat Exchanger Equipment Schedule",
        "| Exchanger | Type | Hot Stream | Cold Stream | Duty (kW) | Hot In/Out (°C) | Cold In/Out (°C) | Area (m²) | Cost ($) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for e in eq:
        etype = "Utility" if e["is_utility"] else "Process"
        hin = f"{e['hot_inlet_c']:.1f}" if e["hot_inlet_c"] is not None else "-"
        hout = f"{e['hot_outlet_c']:.1f}" if e["hot_outlet_c"] is not None else "-"
        cin = f"{e['cold_inlet_c']:.1f}" if e["cold_inlet_c"] is not None else "-"
        cout = f"{e['cold_outlet_c']:.1f}" if e["cold_outlet_c"] is not None else "-"
        area = f"{e['area_m2']:.2f}" if e["area_m2"] is not None else "-"
        cost = f"${e['installed_cost_usd']:,.0f}" if e["installed_cost_usd"] is not None else "-"
        lines.append(f"| {e['name']} | {etype} | {e['hot_stream']} | {e['cold_stream']} | {e['duty_kw']:.2f} | {hin} / {hout} | {cin} / {cout} | {area} | {cost} |")

    lines.extend([
        "",
        "## 4. Techno-Economic & Environmental Summary",
        f"- **Annual OPEX (Integrated):** ${econ['annual_operating_cost_usd_per_yr']:,.2f}/yr",
        f"- **Annual OPEX (Un-integrated):** ${econ['unintegrated_operating_cost_usd_per_yr']:,.2f}/yr" if econ.get("unintegrated_operating_cost_usd_per_yr") else "- **Annual OPEX (Un-integrated):** N/A",
        f"- **Total Annualized Cost (TAC):** ${econ['total_annualized_cost_usd_per_yr']:,.2f}/yr" if econ.get("total_annualized_cost_usd_per_yr") else "- **Total Annualized Cost (TAC):** N/A",
        f"- **Net Present Value (NPV):** ${econ['net_present_value_usd']:,.2f}" if econ.get("net_present_value_usd") else "- **Net Present Value (NPV):** N/A",
        f"- **Carbon Tax Cost Savings:** ${emiss['annual_carbon_cost_savings_usd']:,.2f}/yr" if emiss.get("annual_carbon_cost_savings_usd") else "- **Carbon Tax Cost Savings:** N/A",
    ])

    if report.get("retrofit"):
        ret = report["retrofit"]
        lines.extend([
            "",
            "## 5. Retrofit Recommendations",
            f"- **Optimal:** {'Yes' if ret['is_already_optimal'] else 'No'}",
            f"- **Potential Energy Savings:** {ret['energy_savings_kw']:.2f} kW ({ret['energy_savings_percent']:.1f}%)",
            f"- **Annual Retrofit Savings:** ${ret['annual_cost_savings_usd']:,.2f}/yr",
            f"- **Estimated Retrofit CAPEX:** ${ret['estimated_retrofit_capex_usd']:,.2f}",
            f"- **Retrofit Payback:** {ret['simple_payback_years']:.2f} years" if ret.get("simple_payback_years") else "- **Retrofit Payback:** N/A",
            "",
            "### Recommended Modifications:",
        ])
        for m in ret["modifications"]:
            lines.append(f"- **[{m['action'].upper()}] {m['unit_name']}:** {m['description']} (Est. Cost: ${m['estimated_cost_usd']:,.0f})")

    return "\n".join(lines)


def export_report_html(report: Dict[str, Any], title: str = "PinchPy HEN Analysis Report") -> str:
    """Format report dictionary as a standalone, responsive HTML page with modern styling."""
    meta = report["metadata"]
    st = report["streams"]
    tgt = report["pinch_targets"]
    econ = report["economics"]
    emiss = report["emissions"]
    eq = report["equipment_schedule"]

    capex_str = f"${econ['installed_capital_cost_usd']:,.2f}" if econ.get("installed_capital_cost_usd") else "N/A"
    savings_str = f"${econ['annual_operating_savings_usd_per_yr']:,.2f}/yr" if econ.get("annual_operating_savings_usd_per_yr") else "N/A"
    payback_str = f"{econ['simple_payback_years']:.2f} yrs" if econ.get("simple_payback_years") else "N/A"
    co2_str = f"{emiss['annual_co2_reduction_tonnes']:.1f} t/yr" if emiss.get("annual_co2_reduction_tonnes") else "N/A"

    stream_rows_html = "".join(
        f"<tr><td>{s['id']}</td><td><span class='badge {s['type']}'>{s['type'].upper()}</span></td>"
        f"<td>{s['supply_temperature_c']:.1f}</td><td>{s['target_temperature_c']:.1f}</td>"
        f"<td>{s['cp_kw_per_c']:.2f}</td><td>{s['duty_kw']:.1f}</td></tr>"
        for s in st["stream_list"]
    )

    eq_rows = []
    for e in eq:
        etype = 'Utility' if e['is_utility'] else 'Process'
        badge_cls = 'utility' if e['is_utility'] else 'process'
        hin = f"{e['hot_inlet_c']:.1f} &rarr; {e['hot_outlet_c']:.1f}" if e['hot_inlet_c'] is not None else '-'
        cin = f"{e['cold_inlet_c']:.1f} &rarr; {e['cold_outlet_c']:.1f}" if e['cold_inlet_c'] is not None else '-'
        area_s = f"{e['area_m2']:.2f}" if e['area_m2'] is not None else '-'
        cost_s = f"${e['installed_cost_usd']:,.0f}" if e['installed_cost_usd'] is not None else '-'
        eq_rows.append(
            f"<tr><td><strong>{e['name']}</strong></td>"
            f"<td><span class='badge {badge_cls}'>{etype}</span></td>"
            f"<td>{e['hot_stream']}</td><td>{e['cold_stream']}</td>"
            f"<td>{e['duty_kw']:.1f}</td>"
            f"<td>{hin}</td><td>{cin}</td>"
            f"<td>{area_s}</td><td>{cost_s}</td></tr>"
        )
    eq_rows_html = ''.join(eq_rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 24px; background: #f8fafc; color: #1e293b; }}
  .container {{ max-width: 1050px; margin: 0 auto; background: #ffffff; padding: 36px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }}
  h1 {{ margin-top: 0; color: #0f172a; font-size: 28px; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
  h2 {{ color: #1e293b; font-size: 20px; margin-top: 32px; margin-bottom: 16px; border-left: 4px solid #3b82f6; padding-left: 10px; }}
  .meta {{ color: #64748b; font-size: 14px; margin-bottom: 24px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 28px; }}
  .card {{ background: #f1f5f9; padding: 18px; border-radius: 8px; border-top: 3px solid #3b82f6; }}
  .card .label {{ font-size: 12px; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.5px; }}
  .card .value {{ font-size: 22px; font-weight: 700; color: #0f172a; margin-top: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f8fafc; color: #475569; font-weight: 600; font-size: 13px; text-transform: uppercase; }}
  tr:hover {{ background: #f8fafc; }}
  .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
  .badge.hot {{ background: #fee2e2; color: #dc2626; }}
  .badge.cold {{ background: #e0f2fe; color: #0284c7; }}
  .badge.process {{ background: #dcfce7; color: #16a34a; }}
  .badge.utility {{ background: #f1f5f9; color: #475569; }}
</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <div class="meta">PinchPy v{meta['tool']} | Analyzed on {meta['timestamp'][:19].replace('T', ' ')} | &Delta;T<sub>min</sub> = {meta['delta_t_min']} &deg;C</div>

  <div class="cards">
    <div class="card"><div class="label">Pinch Temperature</div><div class="value">{tgt['pinch_hot_temperature']:.1f} / {tgt['pinch_cold_temperature']:.1f} &deg;C</div></div>
    <div class="card"><div class="label">Min Hot Utility</div><div class="value">{tgt['minimum_hot_utility']:.1f} kW</div></div>
    <div class="card"><div class="label">Min Cold Utility</div><div class="value">{tgt['minimum_cold_utility']:.1f} kW</div></div>
    <div class="card"><div class="label">Heat Recovery</div><div class="value">{tgt.get('heat_recovery', 0.0):.1f} kW</div></div>
  </div>

  <div class="cards">
    <div class="card"><div class="label">Installed CAPEX</div><div class="value">{capex_str}</div></div>
    <div class="card"><div class="label">Annual Savings</div><div class="value">{savings_str}</div></div>
    <div class="card"><div class="label">Simple Payback</div><div class="value">{payback_str}</div></div>
    <div class="card"><div class="label">CO2 Avoided</div><div class="value">{co2_str}</div></div>
  </div>

  <h2>Process Stream Inventory</h2>
  <table>
    <thead><tr><th>ID</th><th>Type</th><th>Supply (&deg;C)</th><th>Target (&deg;C)</th><th>CP (kW/&deg;C)</th><th>Duty (kW)</th></tr></thead>
    <tbody>{stream_rows_html}</tbody>
  </table>

  <h2>Heat Exchanger Network Equipment Schedule</h2>
  <table>
    <thead><tr><th>Exchanger</th><th>Type</th><th>Hot Stream</th><th>Cold Stream</th><th>Duty (kW)</th><th>Hot Range (&deg;C)</th><th>Cold Range (&deg;C)</th><th>Area (m&sup2;)</th><th>Cost ($)</th></tr></thead>
    <tbody>{eq_rows_html}</tbody>
  </table>
</div>
</body>
</html>
"""
    return html


def export_report_json(report: Dict[str, Any], indent: int = 2) -> str:
    """Serialize the report dictionary to a JSON string."""
    return json.dumps(report, indent=indent, default=str)
