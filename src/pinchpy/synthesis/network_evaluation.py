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
