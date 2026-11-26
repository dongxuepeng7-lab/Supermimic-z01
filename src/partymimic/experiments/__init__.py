"""RA-L experiments on the local PartyMimic Z01 implementation."""

GROUPS = {
    "A0": dict(plant="nominal", model="base", response=False, window="fixed"),
    "A": dict(plant="identified", model="base", response=False, window="fixed"),
    "A+": dict(plant="identified", model="matched", response=False, window="fixed"),
    "B": dict(plant="identified", model="actuator", response=False, window="fixed"),
    "C": dict(plant="identified", model="actuator", response=True, window="fixed"),
    "C-perm": dict(
        plant="identified", model="actuator", response=True, window="permuted"
    ),
    "D": dict(plant="identified", model="actuator", response=True, window="calibrated"),
}
