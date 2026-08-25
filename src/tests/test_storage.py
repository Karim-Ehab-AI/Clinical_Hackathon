import os
import shutil
import tempfile
import pytest
from services.storage_service import StorageService


@pytest.fixture
def temp_storage():
    temp_dir = tempfile.mkdtemp()
    service = StorageService(assets_dir=temp_dir)
    yield service, temp_dir
    shutil.rmtree(temp_dir)


def test_calculate_hash_and_save(temp_storage):
    service, temp_dir = temp_storage
    content = b"Sample Clinical PDF Content For Testing"

    file_hash, file_path, already_existed = service.save_file(content)

    assert len(file_hash) == 64  # SHA-256 hex string length
    assert already_existed is False
    assert os.path.exists(file_path)
    assert file_path.endswith(f"{file_hash}.pdf")

    # Second save should detect existing file
    hash2, path2, already_existed2 = service.save_file(content)
    assert hash2 == file_hash
    assert already_existed2 is True
