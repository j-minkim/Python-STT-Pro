import gdown
import os
import re
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


def download_folder_from_gdrive(url, output_dir):
    """
    Download every file from a Google Drive public folder link.
    Returns the list of downloaded paths reported by gdown.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        normalized_url = normalize_gdrive_folder_url(url)
        console.print(f"Downloading Google Drive folder: [cyan]{normalized_url}[/cyan]")
        paths = gdown.download_folder(
            url=normalized_url,
            output=output_dir,
            quiet=False,
            remaining_ok=True,
            resume=True,
        )

        if paths is None:
            raise RuntimeError(
                "Google Drive 폴더 목록을 가져오거나 다운로드하지 못했습니다. "
                "폴더 공유 권한이 '링크가 있는 모든 사용자'인지 확인해 주세요."
            )

        if len(paths) > 0:
            console.print(f"[bold green]Folder download successful:[/bold green] {len(paths)} files")
            return paths

        raise RuntimeError(
            "Google Drive 폴더에서 다운로드할 파일을 찾지 못했습니다. "
            "폴더가 비어 있거나, 파일이 바로가기/권한 제한 상태일 수 있습니다."
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
