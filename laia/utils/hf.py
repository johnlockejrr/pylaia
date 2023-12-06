from huggingface_hub import repo_exists, snapshot_download


def download_from_huggingface(repo_id: str) -> str:
    if not repo_exists(repo_id=repo_id):
        raise Exception(f'HF Repository "{repo_id}" does not exist.')

    return snapshot_download(repo_id=repo_id)
