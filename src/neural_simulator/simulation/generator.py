"""Random but reproducible scenario generation."""

from __future__ import annotations

import random

from neural_simulator.simulation.scenario import Machine, Order, SupplyChainScenario


DEFAULT_PRODUCT_TYPES = ("A", "B", "C")
DEFAULT_DISPATCH_RULES = ("fifo", "earliest_due_date", "shortest_processing_time")


def generate_scenario(
    seed: int,
    *,
    n_machines: int | None = None,
    n_orders: int | None = None,
    product_types: tuple[str, ...] = DEFAULT_PRODUCT_TYPES,
    dispatch_rules: tuple[str, ...] = DEFAULT_DISPATCH_RULES,
    min_route_length: int = 1,
    max_route_length: int | None = None,
    reentrant_route_probability: float = 0.0,
    route_shuffle_probability: float = 0.0,
    wide_input_buffers: bool = False,
) -> SupplyChainScenario:
    """Generate a small supply-chain scenario from a fixed seed."""

    if min_route_length < 1:
        raise ValueError("min_route_length must be positive")
    if max_route_length is not None and max_route_length < min_route_length:
        raise ValueError("max_route_length must be >= min_route_length")
    _validate_probability(
        reentrant_route_probability,
        "reentrant_route_probability",
    )
    _validate_probability(route_shuffle_probability, "route_shuffle_probability")

    rng = random.Random(seed)
    machine_count = n_machines if n_machines is not None else rng.randint(3, 6)
    order_count = n_orders if n_orders is not None else rng.randint(5, 20)
    dispatch_rule = rng.choice(dispatch_rules)
    input_buffer_capacity = order_count if wide_input_buffers else None

    machines = [
        Machine(
            machine_id=f"M{idx}",
            speed=round(rng.uniform(0.8, 1.4), 3),
            capacity=rng.choice([1, 1, 1, 2]),
            input_buffer_capacity=(
                input_buffer_capacity
                if input_buffer_capacity is not None
                else rng.choice([1, 1, 2, 2, 3])
            ),
            processing_time_by_product_type={
                product: round(rng.uniform(0.7, 2.6), 3) for product in product_types
            },
            setup_time_by_product_transition=_generate_setup_matrix(rng, product_types),
        )
        for idx in range(machine_count)
    ]

    machine_ids = [machine.machine_id for machine in machines]
    machine_map = {machine.machine_id: machine for machine in machines}

    orders: list[Order] = []
    for idx in range(order_count):
        product_type = rng.choice(product_types)
        route = _generate_route(
            rng,
            machine_ids,
            min_route_length=min_route_length,
            max_route_length=max_route_length,
            reentrant_route_probability=reentrant_route_probability,
            route_shuffle_probability=route_shuffle_probability,
        )
        quantity = float(rng.randint(1, 10))
        release_time = round(rng.uniform(0.0, 20.0), 3)

        nominal_processing = sum(
            quantity
            * machine_map[machine_id].processing_time_by_product_type[product_type]
            / machine_map[machine_id].speed
            for machine_id in route
        )
        nominal_setup = sum(
            machine_map[machine_id].mean_setup_time_to_product(product_type)
            for machine_id in route
        )
        due_date = round(
            release_time + nominal_processing + nominal_setup + rng.uniform(5.0, 30.0),
            3,
        )

        orders.append(
            Order(
                order_id=f"O{idx}",
                quantity=quantity,
                release_time=release_time,
                due_date=due_date,
                product_type=product_type,
                route=route,
            )
        )

    orders.sort(key=lambda order: (order.release_time, order.order_id))
    return SupplyChainScenario(
        scenario_id=f"scenario-{seed}",
        seed=seed,
        machines=machines,
        orders=orders,
        dispatch_rule=dispatch_rule,
    )


def _generate_setup_matrix(
    rng: random.Random,
    product_types: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    return {
        source: {
            target: 0.0 if source == target else round(rng.uniform(0.4, 3.0), 3)
            for target in product_types
        }
        for source in product_types
    }


def _generate_route(
    rng: random.Random,
    machine_ids: list[str],
    *,
    min_route_length: int,
    max_route_length: int | None,
    reentrant_route_probability: float,
    route_shuffle_probability: float,
) -> list[str]:
    default_max_route_length = min(4, len(machine_ids))
    route_length_upper = max(min_route_length, max_route_length or default_max_route_length)
    route_length = rng.randint(min_route_length, route_length_upper)

    if len(machine_ids) == 1:
        return [machine_ids[0]] * route_length

    needs_reentrant_route = route_length > len(machine_ids)
    if (
        route_length > 2
        and (needs_reentrant_route or rng.random() < reentrant_route_probability)
    ):
        return _generate_reentrant_route(rng, machine_ids, route_length)

    unique_route_length = min(route_length, len(machine_ids))
    route = rng.sample(machine_ids, unique_route_length)
    if rng.random() >= route_shuffle_probability:
        route = sorted(route, key=_machine_sort_key)
    return route


def _generate_reentrant_route(
    rng: random.Random,
    machine_ids: list[str],
    route_length: int,
) -> list[str]:
    route = [rng.choice(machine_ids)]
    while len(route) < route_length:
        candidates = [machine_id for machine_id in machine_ids if machine_id != route[-1]]
        if len(route) >= 2 and rng.random() < 0.45:
            previous_candidates = [
                machine_id for machine_id in route[:-1] if machine_id != route[-1]
            ]
            if previous_candidates:
                candidates = previous_candidates
        route.append(rng.choice(candidates))

    if len(set(route)) == len(route):
        repeat_position = rng.randrange(0, route_length - 2)
        route[-1] = route[repeat_position]
    return route


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _machine_sort_key(machine_id: str) -> int:
    digits = "".join(character for character in machine_id if character.isdigit())
    return int(digits) if digits else 0
