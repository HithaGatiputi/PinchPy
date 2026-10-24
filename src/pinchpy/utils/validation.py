def validate_stream(stream):
    if stream.supply_temperature == stream.target_temperature:
        raise ValueError("Supply and target temperatures cannot be identical.")
    if stream.heat_capacity_flow <= 0:
        raise ValueError("Heat-capacity flow must be positive.")
