import gdown
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlunparse
from rich.console import Console

console = Console()

def download_from_gdrive(url, output_path):
    """
    Download a file from a Google Drive public link.
    """
    try:
        console.print(f"Downloading from Google Drive: [cyan]{url}[/cyan]")
        # gdown.download handles common URL formats automatically
        path = gdown.download(url, output_path, quiet=False)
        
        if path:
            console.print(f"[bold green]Download successful:[/bold green] {path}")
            return path
        else:
            console.print("[red]Download failed.[/red]")
            return None
    except Exception as e:
        console.print(f"[red]Error during download:[/red] {str(e)}")
        return None

def normalize_gdrive_folder_url(url):
    """Return a gdown-friendly Google Drive folder URL."""
    parsed = urlparse(url)
    folder_match = re.search(r"/folders/([^/?#]+)", parsed.path)
    folder_id = None

    if folder_match:
        folder_id = folder_match.group(1)
    else:
        query = parse_qs(parsed.query)
        folder_id = (query.get("id") or [None])[0]

    if not folder_id:
        return url

    query = parse_qs(parsed.query)
    clean_query = ""
    if query.get("resourcekey"):
        clean_query = f"resourcekey={query['resourcekey'][0]}"

    return urlunparse(("https", "drive.google.com", f"/drive/folders/{folder_id}", "", clean_query, ""))


def _download_gdrive_file_by_id(file_id, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return gdown.download(
        url=f"https://drive.google.com/uc?id={file_id}",
        output=output_path,
        quiet=False,
        fuzzy=False,
        resume=True,
    )


def _folder_item_path(item):
    local_path = getattr(item, "local_path", None)
    if local_path:
        return local_path

    path = getattr(item, "path", None)
    if path:
        return path

    return str(item)


def _folder_item_id(item):
    return getattr(item, "id", None)


def list_gdrive_folder_files(url, output_dir):
    normalized_url = normalize_gdrive_folder_url(url)
    console.print(f"Listing Google Drive folder: [cyan]{normalized_url}[/cyan]")
    files = gdown.download_folder(
        url=normalized_url,
        output=output_dir,
        quiet=False,
        remaining_ok=True,
        skip_download=True,
    )

    if files is None:
        raise RuntimeError(
            "Google Drive 폴더 목록을 가져오지 못했습니다. "
            "폴더 공유 권한이 '링크가 있는 모든 사용자'인지 확인해 주세요."
        )

    if len(files) == 0:
        raise RuntimeError(
            "Google Drive 폴더 목록은 열렸지만 파일이 없습니다. "
            "폴더가 비어 있거나 실제 파일 대신 바로가기만 있을 수 있습니다."
        )

    console.print(f"[bold green]Folder listing successful:[/bold green] {len(files)} files")
    return normalized_url, files


def download_folder_from_gdrive(url, output_dir):
    """
    Download every file from a Google Drive public folder link.
    Returns the local paths that were actually downloaded.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        normalized_url, listed_files = list_gdrive_folder_files(url, output_dir)
        console.print(f"Downloading Google Drive folder files: [cyan]{normalized_url}[/cyan]")

        downloaded_paths = []
        failed_names = []
        output_root = Path(output_dir).resolve()

        for index, item in enumerate(listed_files, 1):
            file_id = _folder_item_id(item)
            local_path = Path(_folder_item_path(item))
            if not local_path.is_absolute():
                local_path = output_root / local_path

            display_name = getattr(item, "path", None) or local_path.name
            if not file_id:
                failed_names.append(display_name)
                console.print(f"[yellow]Skipping item without file id:[/yellow] {display_name}")
                continue

            console.print(f"[{index}/{len(listed_files)}] Downloading: [cyan]{display_name}[/cyan]")
            downloaded = _download_gdrive_file_by_id(file_id, str(local_path))
            if downloaded:
                downloaded_paths.append(downloaded)
            else:
                failed_names.append(display_name)

        if downloaded_paths:
            console.print(
                f"[bold green]Folder download successful:[/bold green] "
                f"{len(downloaded_paths)}/{len(listed_files)} files"
            )
            return downloaded_paths

        raise RuntimeError(
            "Google Drive 폴더 목록은 열렸지만 파일 다운로드가 모두 실패했습니다. "
            f"실패 파일: {', '.join(failed_names[:10])}"
        )
    except Exception as e:
        console.print(f"[red]Error during folder download:[/red] {str(e)}")
        raise

def is_gdrive_url(url):
    """Check if the URL is a Google Drive link."""
    return "drive.google.com" in url

def is_gdrive_folder_url(url):
    """Check if the URL points to a Google Drive folder."""
    return bool(url and "drive.google.com" in url and re.search(r"/folders/", url))
