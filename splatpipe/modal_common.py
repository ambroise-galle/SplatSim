"""Ressources Modal partagées par toutes les étapes de SplatPipeline."""

import modal

APP_PREFIX = "splatsim"

# Volume v2 : pas de limite sur le nombre de fichiers (v1 ralentit au-delà de 50 000).
DATA_VOLUME_NAME = "splatsim-data"
DATA_MOUNT = "/data"


def data_volume() -> modal.Volume:
    return modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True, version=2)
