def set_curriculum(cfg: dict, phase: int):
    if phase == 1:
        cfg["enemy_penalty"]["density_weight"] = 0.40
        cfg["idle_penalty"]["weight"] = 0.06
        cfg["_xp_scale"] = 1.2
    elif phase == 2:
        cfg["enemy_penalty"]["density_weight"] = 0.25
        cfg["idle_penalty"]["weight"] = 0.03
        cfg["_xp_scale"] = 2.0
    else:
        cfg["enemy_penalty"]["density_weight"] = 0.18
        cfg["idle_penalty"]["weight"] = 0.02
        cfg["_xp_scale"] = 2.6
