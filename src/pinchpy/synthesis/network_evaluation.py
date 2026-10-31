"""Quality metrics for a synthesized heat exchanger network (Phase 6)."""


def evaluate_network(network, streams=None):
    """Summarize a ``HeatExchangerNetwork``.

    Always returns ``number_of_exchangers`` and ``total_recovered_heat``
    (process-to-process duty only, excluding utility exchangers when they are
    flagged via ``is_utility``). When ``streams`` is given, also returns
    ``minimum_units`` -- the theoretical lower bound on the number of units
    for a loop-free network, ``N_streams + N_utility_types - 1`` -- for
    comparison against the actual exchanger count.
    """
    exchangers = network.exchangers
    process_exchangers = [e for e in exchangers if not getattr(e, "is_utility", False)]
    utility_exchangers = [e for e in exchangers if getattr(e, "is_utility", False)]

    result = {
        "number_of_exchangers": len(exchangers),
        "number_of_process_matches": len(process_exchangers),
        "number_of_utility_exchangers": len(utility_exchangers),
        "total_recovered_heat": sum(e.duty for e in process_exchangers),
        "total_utility_heat": sum(e.duty for e in utility_exchangers),
    }

    areas = [e.area for e in exchangers if e.area is not None]
    if areas:
        result["total_area"] = sum(areas)

    if streams is not None:
        n_streams = len(streams)
        n_utility_types = len({e.hot_stream if e.hot_stream in ("HU", "CU") else e.cold_stream
                                for e in utility_exchangers})
        result["minimum_units"] = max(n_streams + n_utility_types - 1, 0)

    return result


def network_tac(
    network,
    fixed_cost: float = 2000.0,
    area_cost: float = 800.0,
    area_exponent: float = 0.8,
    hot_utility_cost: float = 80.0,
    cold_utility_cost: float = 15.0,
) -> dict[str, float]:
    """Calculate the Total Annual Cost (TAC) of a synthesized heat exchanger network.

    Computes capital and operational expenditure following the formulation in
    Yang et al., Energy 324 (2025) 136011 (Eq. 1):

        TAC = sum over exchangers (fixed_cost + area_cost * area**area_exponent)
              + hot_utility_cost * total_heater_duty
              + cold_utility_cost * total_cooler_duty

    Parameters
    ----------
    network : HeatExchangerNetwork
        Synthesized heat exchanger network containing process and utility exchangers.
    fixed_cost : float, default 2000.0
        Annualized fixed installation cost per exchanger unit ($/yr).
    area_cost : float, default 800.0
        Annualized area cost coefficient ($/(yr·(m²)^area_exponent)).
    area_exponent : float, default 0.8
        Cost economy-of-scale area exponent (dimensionless, typically 0.6–1.0).
    hot_utility_cost : float, default 80.0
        Annualized cost rate of hot utility per kilowatt of duty ($/(kW·yr)).
    cold_utility_cost : float, default 15.0
        Annualized cost rate of cold utility per kilowatt of duty ($/(kW·yr)).

    Returns
    -------
    dict[str, float]
        Cost breakdown dictionary with:
        - 'capital': Annualized capital cost of all exchangers ($/yr).
        - 'hot_utility': Annual operating cost for hot utility ($/yr).
        - 'cold_utility': Annual operating cost for cold utility ($/yr).
        - 'total': Total Annual Cost ($/yr).

    Raises
    ------
    ValueError
        If any exchanger in the network has ``area is None`` (i.e., overall heat
        transfer coefficient U was not provided during network synthesis).
    """
    capital_cost = 0.0
    heater_duty = 0.0
    cooler_duty = 0.0

    for e in network.exchangers:
        if e.area is None:
            raise ValueError(
                f"Exchanger '{e.name}' has area None. Total annual cost (TAC) calculation "
                "requires all exchanger areas (provide overall_heat_transfer_coefficient U)."
            )
        capital_cost += fixed_cost + area_cost * (float(e.area) ** area_exponent)

        if getattr(e, "is_utility", False):
            if e.hot_stream in ("HU", "hot_utility") or e.name.upper().startswith("HU") or e.cold_stream != "CU":
                heater_duty += float(e.duty)
            elif e.cold_stream in ("CU", "cold_utility") or e.name.upper().startswith("CU") or e.hot_stream != "HU":
                cooler_duty += float(e.duty)
        else:
            if e.hot_stream in ("HU", "hot_utility"):
                heater_duty += float(e.duty)
            elif e.cold_stream in ("CU", "cold_utility"):
                cooler_duty += float(e.duty)

    hot_util_cost = hot_utility_cost * heater_duty
    cold_util_cost = cold_utility_cost * cooler_duty
    total_tac = capital_cost + hot_util_cost + cold_util_cost

    return {
        "capital": capital_cost,
        "hot_utility": hot_util_cost,
        "cold_utility": cold_util_cost,
        "total": total_tac,
    }

