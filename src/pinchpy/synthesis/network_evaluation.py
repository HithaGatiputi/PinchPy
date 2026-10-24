def evaluate_network(network):
    return {
        "number_of_exchangers": len(network.exchangers),
        "total_recovered_heat": network.total_recovered_heat(),
    }
