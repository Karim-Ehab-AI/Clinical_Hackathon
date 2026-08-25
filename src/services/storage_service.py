import hashlib
import os
from typing import Tuple
from config.settings import settings


class StorageService:
    """Handles file saving, SHA-256 content hashing, and duplicate detection in assets directory."""

    def __init__(self, assets_dir: str = settings.ASSETS_DIR):
        self.assets_dir = assets_dir
        os.makedirs(self.assets_dir, exist_ok=True)

    def calculate_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of file byte content."""
        return hashlib.sha256(content).hexdigest()

    def get_file_path(self, file_hash: str) -> str:
        """Get target file path for a given file hash."""
        return os.path.join(self.assets_dir, f"{file_hash}.pdf")

    def exists(self, file_hash: str) -> bool:
        """Check if file with given hash exists in assets directory."""
        path = self.get_file_path(file_hash)
        return os.path.exists(path)

    def save_file(self, content: bytes) -> Tuple[str, str, bool]:
        """Save file content named by its SHA-256 hash to assets directory.
        Returns (file_hash, file_path, already_existed).
        """
        file_hash = self.calculate_hash(content)
        file_path = self.get_file_path(file_hash)
        already_existed = os.path.exists(file_path)

        if not already_existed:
            with open(file_path, "wb") as f:
                f.write(content)

        return file_hash, file_path, already_existed
