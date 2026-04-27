"""File utilities for VFI-gui.

Provides file sorting, image sequence detection, and media type utilities.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Pattern

from loguru import logger


# Supported image extensions
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".exr",
    ".bmp", ".webp", ".gif", ".dds", ".hdr",
}

# Supported video extensions
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".webm", ".m4v", ".mpg", ".mpeg", ".ts", ".m2ts",
    ".vob", ".ogv", ".rm", ".rmvb", ".asf",
}

# Common image sequence patterns
# Examples: frame_0001.png, img001.jpg, 0001.exr, scene1_frame0001.png
SEQUENCE_PATTERNS = [
    # Pattern with separator and padding: name_0001.ext
    re.compile(r"^(.+?)[_.-](\d+)(\.\w+)$"),
    # Pattern with only numbers: 0001.ext
    re.compile(r"^(\d+)(\.\w+)$"),
    # Pattern with prefix and numbers: img0001.ext
    re.compile(r"^([a-zA-Z]+)(\d+)(\.\w+)$"),
]


def natural_sort_key(text: str) -> List:
    """Generate a natural sort key for alphanumeric sorting.
    
    This allows sorting strings like: file1, file2, file10, file20
    instead of: file1, file10, file2, file20
    
    Args:
        text: String to generate sort key for
        
    Returns:
        List of mixed string/int parts for sorting
        
    Example:
        >>> sorted(['file10', 'file2', 'file1'], key=natural_sort_key)
        ['file1', 'file2', 'file10']
    """
    def convert(part: str):
        return int(part) if part.isdigit() else part.lower()
    
    return [convert(c) for c in re.split(r'(\d+)', str(text))]


def sort_files_naturally(files: List[Path]) -> List[Path]:
    """Sort files using natural (alphanumeric) sorting.
    
    Args:
        files: List of file paths to sort
        
    Returns:
        Sorted list of file paths
    """
    return sorted(files, key=lambda f: natural_sort_key(f.name))


def is_image_file(path: Path) -> bool:
    """Check if file is a supported image.
    
    Args:
        path: File path to check
        
    Returns:
        True if file has a supported image extension
    """
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: Path) -> bool:
    """Check if file is a supported video.
    
    Args:
        path: File path to check
        
    Returns:
        True if file has a supported video extension
    """
    return path.suffix.lower() in VIDEO_EXTENSIONS


def get_media_type(path: Path) -> Optional[str]:
    """Get media type of file.
    
    Args:
        path: File path to check
        
    Returns:
        'image', 'video', or None if not recognized
    """
    if is_image_file(path):
        return "image"
    elif is_video_file(path):
        return "video"
    return None


def detect_image_sequence_pattern(
    files: List[Path]
) -> Tuple[Optional[Pattern], Optional[str], int, Optional[str]]:
    """Detect the naming pattern of an image sequence.
    
    Analyzes file names to find a common pattern with sequential numbers.
    
    Args:
        files: List of image file paths
        
    Returns:
        Tuple of (compiled_pattern, prefix, padding_digits, separator)
        - pattern: Compiled regex pattern
        - prefix: Prefix string (without separator)
        - padding_digits: Number of digits for frame number
        - separator: Separator between prefix and number ('_', '.', '-', or '')
        Returns (None, None, 0, None) if no pattern detected
        
    Example:
        >>> files = [Path('frame_0001.png'), Path('frame_0002.png')]
        >>> pattern, prefix, padding, sep = detect_image_sequence_pattern(files)
        >>> # pattern matches 'frame_0001.png', prefix='frame', padding=4, sep='_'
    """
    if not files:
        return None, None, 0, None
    
    # Try each pattern type
    for pattern in SEQUENCE_PATTERNS:
        prefixes = set()
        paddings = set()
        separators = set()
        matches = 0
        
        for f in files:
            match = pattern.match(f.name)
            if match:
                matches += 1
                groups = match.groups()
                
                if len(groups) == 2:
                    # Pattern: 0001.ext
                    prefix = ""
                    number = groups[0]
                    separator = ""
                elif len(groups) == 3:
                    # Pattern: prefix_0001.ext or prefix0001.ext
                    prefix = groups[0]
                    number = groups[1]
                    # Detect separator from original filename
                    # Find what comes between prefix and number
                    rest = f.name[len(prefix):]
                    if rest and rest[0] in '_.-':
                        separator = rest[0]
                    else:
                        separator = ""
                else:
                    continue
                
                prefixes.add(prefix)
                paddings.add(len(number))
                separators.add(separator)
        
        # Check if pattern matches most files
        if matches >= len(files) * 0.8 and len(prefixes) == 1:
            return (
                pattern,
                prefixes.pop(),
                max(paddings),
                separators.pop() if len(separators) == 1 else "",
            )
    
    return None, None, 0, None


def get_image_sequence_files(
    path: Path,
    extensions: Optional[List[str]] = None,
    sort: bool = True,
) -> List[Path]:
    """Get all image files from a path (file or directory).
    
    If path is a file, returns a single-item list if it's an image.
    If path is a directory, returns all image files sorted naturally.
    
    Args:
        path: File or directory path
        extensions: Optional list of extensions to filter (e.g., ['.png', '.jpg'])
        sort: Whether to sort files naturally (default: True)
        
    Returns:
        List of image file paths, sorted if requested
        
    Raises:
        ValueError: If path doesn't exist
    """
    path = Path(path)
    
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    
    files = []
    
    if path.is_file():
        # Single file
        if is_image_file(path):
            if extensions is None or path.suffix.lower() in extensions:
                files.append(path)
        else:
            logger.warning(f"Not an image file: {path}")
    
    elif path.is_dir():
        # Directory - get all image files
        for ext in extensions or list(IMAGE_EXTENSIONS):
            files.extend(path.glob(f"*{ext}"))
            # Also check uppercase
            files.extend(path.glob(f"*{ext.upper()}"))
        
        # Remove duplicates
        files = list(set(files))
        
        if sort:
            files = sort_files_naturally(files)
    
    logger.debug(f"Found {len(files)} image files in {path}")
    return files


def parse_frame_number(filename: str) -> Optional[int]:
    """Extract frame number from filename.
    
    Tries to extract the last sequence of digits as the frame number.
    
    Args:
        filename: Filename to parse
        
    Returns:
        Frame number or None if not found
        
    Example:
        >>> parse_frame_number("frame_0001.png")
        1
        >>> parse_frame_number("img001.jpg")
        1
        >>> parse_frame_number("scene1_frame0042.exr")
        42
    """
    # Find all digit sequences
    matches = re.findall(r'\d+', filename)
    
    if matches:
        # Return the last match (usually the frame number)
        return int(matches[-1])
    
    return None


def generate_sequence_filename(
    prefix: str,
    frame_number: int,
    extension: str,
    padding: int = 4,
    separator: str = "_",
) -> str:
    """Generate a filename for an image sequence.
    
    Args:
        prefix: Filename prefix (e.g., 'frame', 'img')
        frame_number: Frame number
        extension: File extension (e.g., '.png')
        padding: Number of digits for frame number (default: 4)
        separator: Separator between prefix and number (default: '_')
        
    Returns:
        Generated filename
        
    Example:
        >>> generate_sequence_filename("frame", 1, ".png", 4)
        'frame_0001.png'
    """
    # Ensure extension starts with dot
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    # Format frame number with padding
    frame_str = str(frame_number).zfill(padding)
    
    if prefix:
        return f"{prefix}{separator}{frame_str}{extension}"
    else:
        return f"{frame_str}{extension}"


def validate_image_sequence(
    files: List[Path],
    check_continuity: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Validate an image sequence.
    
    Checks that files form a valid sequence with consistent naming.
    
    Args:
        files: List of image file paths
        check_continuity: Whether to check for missing frames
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not files:
        return False, "No files provided"
    
    # Check all are image files
    for f in files:
        if not is_image_file(f):
            return False, f"Not an image file: {f}"
    
    # Check consistent extension
    extensions = set(f.suffix.lower() for f in files)
    if len(extensions) > 1:
        return False, f"Mixed extensions: {extensions}"
    
    # Check continuity if requested
    if check_continuity and len(files) > 1:
        frame_numbers = []
        for f in files:
            num = parse_frame_number(f.name)
            if num is None:
                return False, f"Cannot parse frame number: {f.name}"
            frame_numbers.append(num)
        
        frame_numbers.sort()
        
        # Check for gaps
        for i in range(1, len(frame_numbers)):
            if frame_numbers[i] - frame_numbers[i-1] > 1:
                logger.warning(
                    f"Frame gap detected: {frame_numbers[i-1]} -> {frame_numbers[i]}"
                )
                # Don't fail on gaps, just warn
    
    return True, None
