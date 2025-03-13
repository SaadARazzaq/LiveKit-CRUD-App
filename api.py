import os
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Annotated
from livekit.agents import llm
import traceback

logger = logging.getLogger("file-tools")
logger.setLevel(logging.INFO)

class AssistantFnc(llm.FunctionContext):
    def __init__(self) -> None:
        super().__init__()
        self.scratch_dir = Path(os.getenv("SCRATCH_PAD_DIR", "./scratchpad")).resolve()
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Scratchpad directory initialized at: {self.scratch_dir}")

    def _validate_path(self, relative_path: str) -> Path:
        """Validate and normalize a relative path"""
        try:
            clean_path = Path(relative_path.strip())
            if not clean_path.parts:
                raise ValueError("Empty path is invalid")
                
            full_path = (self.scratch_dir / clean_path).resolve()
            
            if not full_path.is_relative_to(self.scratch_dir.resolve()):
                raise ValueError("Path traversal attempt detected")
                
            return full_path
        except Exception as e:
            logger.error(f"Path validation failed: {traceback.format_exc()}")
            raise

    # File Operations
    @llm.ai_callable(description="Create a new text file with specified content")
    def create_file(
        self,
        file_path: Annotated[str, llm.TypeInfo(description="Relative path including subdirectories")],
        content: Annotated[str, llm.TypeInfo(description="Text content to write to the file")],
    ) -> str:
        try:
            full_path = self._validate_path(file_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if full_path.exists():
                return f"File {file_path} already exists"
                
            full_path.write_text(content, encoding='utf-8')
            return f"Created {file_path}"
        except Exception as e:
            logger.error(f"Create failed: {traceback.format_exc()}")
            return f"Error creating {file_path}: {str(e)}"

    @llm.ai_callable(description="List all files (without extensions)")
    def list_files(self) -> str:
        try:
            file_list = []
            for path in self.scratch_dir.rglob("*"):
                if path.is_file():
                    rel_path = path.relative_to(self.scratch_dir)
                    file_list.append(str(rel_path.with_suffix('')))
                    
            return "\n".join(sorted(file_list)) if file_list else "No files found"
        except Exception as e:
            logger.error(f"List failed: {traceback.format_exc()}")
            return "Error listing files"

    @llm.ai_callable(description="List all files with full paths and extensions")
    def list_files_with_extensions(self) -> str:
        try:
            file_list = []
            for path in self.scratch_dir.rglob("*"):
                if path.is_file():
                    file_list.append(str(path.relative_to(self.scratch_dir)))
                    
            return "\n".join(sorted(file_list)) if file_list else "No files found"
        except Exception as e:
            logger.error(f"List failed: {traceback.format_exc()}")
            return "Error listing files"

    @llm.ai_callable(description="Read contents of a specific file including subdirectories")
    def read_file(
        self,
        file_path: Annotated[str, llm.TypeInfo(description="Relative path of the file to read")],
    ) -> str:
        try:
            full_path = self._validate_path(file_path)
            if not full_path.exists():
                return f"File {file_path} not found."
            if not full_path.is_file():
                return f"Path {file_path} is not a file."
            
            return f"Contents of {file_path}:\n{full_path.read_text(encoding='utf-8')}"
        except Exception as e:
            logger.error(f"Read file failed: {str(e)}")
            return f"Could not read {file_path}"

    # Folder Operations
    @llm.ai_callable(description="Create a new folder with optional initial files")
    def create_folder(
        self,
        folder_path: Annotated[str, llm.TypeInfo(description="Relative folder path including subdirectories")],
        overwrite: Annotated[bool, llm.TypeInfo(description="Overwrite existing folder")] = False,
    ) -> str:
        try:
            full_path = self._validate_path(folder_path)
            if full_path.exists():
                if full_path.is_file():
                    return f"Path {folder_path} is a file, cannot create folder here."
                if not overwrite:
                    return f"Folder {folder_path} already exists"
                shutil.rmtree(full_path)
            
            full_path.mkdir(parents=True)
            return f"Created folder {folder_path}"
        except Exception as e:
            logger.error(f"Folder creation failed: {str(e)}")
            return f"Failed to create folder {folder_path}"

    @llm.ai_callable(description="List all files and folders with type indicators")
    def list_all(self) -> str:
        """
        Returns a list of all files and folders in the scratchpad directory.
        Directories are marked with (Folder) at the end of their name.
        """
        try:
            items = []
            # Use rglob to recursively list all items
            for path in self.scratch_dir.rglob("*"):
                # Get relative path
                rel_path = path.relative_to(self.scratch_dir)
                
                if path.is_dir():
                    items.append(f"{rel_path} (Folder)")
                elif path.is_file():
                    items.append(str(rel_path))

            # Add root directory items (rglob doesn't include self.scratch_dir itself)
            for path in self.scratch_dir.glob("*"):
                if path.is_dir() and path not in items:
                    rel_path = path.relative_to(self.scratch_dir)
                    items.append(f"{rel_path} (Folder)")
                elif path.is_file() and path not in items:
                    rel_path = path.relative_to(self.scratch_dir)
                    items.append(str(rel_path))

            # Sort alphabetically with folders first
            sorted_items = sorted(items, key=lambda x: (not x.endswith("(Folder)"), x))
            
            return "\n".join(sorted_items) if sorted_items else "No files or folders found"
            
        except Exception as e:
            logger.error(f"List all failed: {traceback.format_exc()}")
            return "Error listing files and folders"

    @llm.ai_callable(description="Delete a folder and its contents")
    def delete_folder(
        self,
        folder_path: Annotated[str, llm.TypeInfo(description="Relative path of folder to delete")],
    ) -> str:
        try:
            full_path = self._validate_path(folder_path)
            if not full_path.exists():
                return f"Folder {folder_path} not found"
            if not full_path.is_dir():
                return f"Path {folder_path} is not a directory"
            
            shutil.rmtree(full_path)
            return f"Deleted folder {folder_path}"
        except Exception as e:
            logger.error(f"Folder deletion failed: {str(e)}")
            return f"Failed to delete folder {folder_path}"

    @llm.ai_callable(description="Rename or move a file/folder")
    def rename_file(
        self,
        old_path: Annotated[str, llm.TypeInfo(description="Current relative path")],
        new_path: Annotated[str, llm.TypeInfo(description="New relative path")],
    ) -> str:
        try:
            src = self._validate_path(old_path)
            dest = self._validate_path(new_path)
            
            if not src.exists():
                return f"Error: Source path '{old_path}' does not exist"
            if dest.exists():
                return f"Error: Destination path '{new_path}' already exists"
                
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            
            return f"Successfully renamed/moved '{old_path}' to '{new_path}'"
        except Exception as e:
            logger.error(f"Rename failed: {traceback.format_exc()}")
            return f"Failed to rename '{old_path}': {str(e)}"

    @llm.ai_callable(description="Update content of an existing file")
    def update_file(
        self,
        file_path: Annotated[str, llm.TypeInfo(description="Relative path of file to update")],
        new_content: Annotated[str, llm.TypeInfo(description="New content to write")],
    ) -> str:
        try:
            full_path = self._validate_path(file_path)
            if not full_path.exists():
                return f"File {file_path} not found"
            if not full_path.is_file():
                return f"Path {file_path} is not a file"
            
            full_path.write_text(new_content, encoding='utf-8')
            return f"Updated {file_path}"
        except Exception as e:
            logger.error(f"Update failed: {str(e)}")
            return f"Failed to update {file_path}"

    @llm.ai_callable(description="Delete a specific file")
    def delete_file(
        self,
        file_path: Annotated[str, llm.TypeInfo(description="Relative path of file to delete")],
    ) -> str:
        try:
            full_path = self._validate_path(file_path)
            if not full_path.exists():
                return f"File {file_path} not found"
            if not full_path.is_file():
                return f"Path {file_path} is not a file"
            
            full_path.unlink()
            return f"Deleted {file_path}"
        except Exception as e:
            logger.error(f"Deletion failed: {str(e)}")
            return f"Failed to delete {file_path}"

    # Utility
    @llm.ai_callable(description="Get current date and time")
    def get_time(self) -> str:
        return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")