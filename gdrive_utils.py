import gdown
import os
import re
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

def download_folder_from_gdrive(url, output_dir):
    """
    Download every file from a Google Drive public folder link.
    Returns the list of downloaded paths reported by gdown.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        console.print(f"Downloading Google Drive folder: [cyan]{url}[/cyan]")
        paths = gdown.download_folder(
            url=url,
            output=output_dir,
            quiet=False,
            remaining_ok=True,
            resume=True,
        )

        if paths:
            console.print(f"[bold green]Folder download successful:[/bold green] {len(paths)} files")
            return paths

        console.print("[red]Folder download failed.[/red]")
        return []
    except Exception as e:
        console.print(f"[red]Error during folder download:[/red] {str(e)}")
        return []

def is_gdrive_url(url):
    """Check if the URL is a Google Drive link."""
    return "drive.google.com" in url

def is_gdrive_folder_url(url):
    """Check if the URL points to a Google Drive folder."""
    return bool(url and "drive.google.com" in url and re.search(r"/folders/", url))
