from app.config import Settings
from app.storage.base import ObjectStorage
from app.storage.gcs import GCSStorage
from app.storage.local import LocalStorage

_storage_instance: ObjectStorage | None = None


def get_storage(settings: Settings | None = None) -> ObjectStorage:
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance

    if settings is None:
        settings = Settings.from_environment()

    if settings.runtime_profile == "local":
        _storage_instance = LocalStorage(base_dir=settings.local_storage_root)
    elif settings.runtime_profile == "cloud-data":
        # Settings validates the required bucket before the provider is constructed.
        _storage_instance = GCSStorage(
            bucket_name=settings.gcs_bucket_name,
            project_id=settings.google_cloud_project,
        )
    elif settings.gcs_bucket_name:
        # Preserve the legacy automatic profile until callers opt into an explicit profile.
        _storage_instance = GCSStorage(
            bucket_name=settings.gcs_bucket_name,
            project_id=settings.google_cloud_project,
        )
    else:
        _storage_instance = LocalStorage()
    return _storage_instance


def set_storage(storage: ObjectStorage | None) -> None:
    """Override storage instance (useful for unit testing)."""
    global _storage_instance
    _storage_instance = storage
