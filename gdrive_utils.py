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
        # gdown.download can handle fuzzy URLs and extracting the ID
        path = gdown.download(url, output_path, quiet=False, fuzzy=True)
        
        if path:
            console.print(f"[bold green]Download successful:[/bold green] {path}")
            return path
        else:
            console.print("[red]Download failed.[/red]")
            return None
    except Exception as e:
        console.print(f"[red]Error during download:[/red] {str(e)}")
        return None

def is_gdrive_url(url):
    """Check if the URL is a Google Drive link."""
    return "drive.google.com" in url
