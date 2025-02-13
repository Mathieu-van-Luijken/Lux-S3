# from src.luxai_s3.env import EnvObs


def calculate_reward(envobs) -> float:
    """
    Calculate the reward based on the environment observations.

    Parameters:
    envobs (EnvObs): An instance of EnvObs containing the environment observations.

    Returns:
    float: The calculated reward.
    """

    def process_relic_nodes(relic_nodes):
        """
        Process the relic nodes positions to calculate a reward component.

        Parameters:
        relic_nodes (chex.Array): Position of all relic nodes with shape (N, 2) for N max relic nodes and 2 features for position (x, y). Number is -1 if not visible.

        Returns:
        float: The reward component from the relic nodes positions.
        """
        # Example processing: count the number of visible relic nodes
        return sum(100 for node in relic_nodes if node[0] != -1 and node[1] != -1)

    # def process_units(units):
    #     """
    #     Process the units to calculate a reward component.

    #     Parameters:
    #     units (UnitState): The state of units in the environment.

    #     Returns:
    #     float: The reward component from the units.
    #     """
    #     # Example processing: sum the energy of all units
    #     return sum(unit[2] for team in units for unit in team)

    def process_team_points(team_points):
        """
        Process the team points to calculate a reward component.

        Parameters:
        team_points (chex.Array): The points of each team.

        Returns:
        float: The reward component from the team points.
        """

        # Example processing: sum the points of all teams
        return sum(team_points * 10)

    def process_map_features(map_features):
        """
        Process the map features to calculate a reward component.

        Parameters:
        map_features (chex.Array): The features of the map.

        Returns:
        float: The reward component from the map features.
        """
        energy = map_features["energy"]
        tile_type = map_features["tile_type"]
        # Example processing: sum the features of the map
        return sum(1 for row in tile_type for value in row if value > -1)

    # Define a list of reward processing functions
    reward_functions = [
        # process_units,
        process_team_points,
        process_map_features,
        process_relic_nodes,
    ]

    # Calculate the total reward by applying each function
    total_reward = sum(
        func(envobs[func.__name__.replace("process_", "")]) for func in reward_functions
    )

    return total_reward
