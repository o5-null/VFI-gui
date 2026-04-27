"""Internationalization (i18n) module for VFI-gui.

Uses Python's gettext module for industry-standard translation management.
Supports runtime language switching and automatic locale detection.
"""

import gettext
import locale
import os
import sys
from pathlib import Path
from typing import Optional, Callable, List
from functools import lru_cache

from PyQt6.QtCore import QObject, pyqtSignal


# Type alias for translation function
TranslationFunc = Callable[[str], str]


class I18NManager(QObject):
    """Manages internationalization for the application.
    
    Features:
    - Runtime language switching
    - Automatic system locale detection
    - Fallback to English for missing translations
    - Signal emission on language change
    
    Usage:
        i18n = I18NManager()
        i18n.init(locales_dir="locales")
        print(i18n.tr("Hello World"))  # Returns translated string
        i18n.set_language("zh_CN")     # Switch to Chinese
    """
    
    # Signal emitted when language changes
    language_changed = pyqtSignal(str)  # New language code
    
    # Supported languages with display names
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "zh_CN": "简体中文",
        "zh_TW": "繁體中文",
    }
    
    # Default language when detection fails
    DEFAULT_LANGUAGE = "en"
    
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._current_language: str = self.DEFAULT_LANGUAGE
        self._translation: Optional[gettext.NullTranslations] = None
        self._locales_dir: Optional[Path] = None
        self._initialized: bool = False
    
    def init(self, locales_dir: Optional[str] = None) -> None:
        """Initialize the i18n system.
        
        Args:
            locales_dir: Path to locales directory. If None, uses paths.locales_dir
        """
        if locales_dir:
            self._locales_dir = Path(locales_dir)
        else:
            # Use centralized path manager
            from core.paths import paths
            self._locales_dir = paths.locales_dir
        
        # Ensure locales directory exists
        self._locales_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect system language
        system_lang = self._detect_system_language()
        
        # Load initial language
        self._load_translation(system_lang)
        self._initialized = True
    
    def _detect_system_language(self) -> str:
        """Detect the system's preferred language.
        
        Returns:
            Language code (e.g., "en", "zh_CN")
        """
        try:
            # Get system locale
            sys_locale = locale.getdefaultlocale()[0]
            if sys_locale:
                # Normalize locale code
                lang = sys_locale.replace("-", "_")
                
                # Check if we support this language
                if lang in self.SUPPORTED_LANGUAGES:
                    return lang
                
                # Try language only (e.g., "zh" from "zh_CN")
                lang_only = lang.split("_")[0]
                for supported in self.SUPPORTED_LANGUAGES:
                    if supported.startswith(lang_only):
                        return supported
        except Exception:
            pass
        
        return self.DEFAULT_LANGUAGE
    
    def _load_translation(self, language: str) -> bool:
        """Load translation for a specific language.
        
        Args:
            language: Language code (e.g., "en", "zh_CN")
            
        Returns:
            True if translation loaded successfully
        """
        if not self._locales_dir:
            return False
        
        try:
            if language == "en":
                # English is the source language, use null translation
                self._translation = gettext.NullTranslations()
            else:
                # Load MO file from locales/{language}/LC_MESSAGES/messages.mo
                mo_path = self._locales_dir / language / "LC_MESSAGES" / "messages.mo"
                
                if mo_path.exists():
                    # Use custom translation class that handles UTF-8 properly
                    self._translation = self._load_mo_file(mo_path)
                else:
                    # Try PO file and compile (development mode)
                    po_path = self._locales_dir / language / "LC_MESSAGES" / "messages.po"
                    if po_path.exists():
                        # Fall back to null translation, MO file needs compilation
                        print(f"Warning: Found .po file but no .mo file at {mo_path}")
                        print(f"Run: python core/po_to_mo.py")
                        self._translation = gettext.NullTranslations()
                    else:
                        # No translation file found
                        self._translation = gettext.NullTranslations()
            
            self._current_language = language
            return True
            
        except Exception as e:
            print(f"Error loading translation for {language}: {e}")
            self._translation = gettext.NullTranslations()
            return False
    
    def _load_mo_file(self, mo_path: Path) -> gettext.NullTranslations:
        """Load a MO file with proper UTF-8 handling.
        
        Args:
            mo_path: Path to the .mo file
            
        Returns:
            GNUTranslations object
        """
        import struct
        
        with open(mo_path, "rb") as f:
            mo_data = f.read()
        
        # Parse MO file
        magic = struct.unpack_from("<I", mo_data, 0)[0]
        if magic != 0x950412DE:
            raise ValueError(f"Invalid MO file magic: {hex(magic)}")
        
        num_strings = struct.unpack_from("<I", mo_data, 8)[0]
        orig_table_offset = struct.unpack_from("<I", mo_data, 12)[0]
        trans_table_offset = struct.unpack_from("<I", mo_data, 16)[0]
        
        # Build translation dictionary
        translations = {}
        for i in range(num_strings):
            # Read original string info
            orig_length = struct.unpack_from("<I", mo_data, orig_table_offset + i * 8)[0]
            orig_offset = struct.unpack_from("<I", mo_data, orig_table_offset + i * 8 + 4)[0]
            
            # Read translation string info
            trans_length = struct.unpack_from("<I", mo_data, trans_table_offset + i * 8)[0]
            trans_offset = struct.unpack_from("<I", mo_data, trans_table_offset + i * 8 + 4)[0]
            
            # Extract strings
            orig = mo_data[orig_offset:orig_offset + orig_length].decode("utf-8")
            trans = mo_data[trans_offset:trans_offset + trans_length].decode("utf-8")
            
            if orig:  # Skip empty keys (header)
                translations[orig] = trans
        
        # Create a custom translations object
        class UTF8Translations(gettext.NullTranslations):
            def __init__(self, catalog):
                super().__init__()
                self._catalog = catalog
            
            def gettext(self, message):
                return self._catalog.get(message, message)
            
            def ngettext(self, singular, plural, n):
                if n == 1:
                    return self._catalog.get(singular, singular)
                return self._catalog.get(plural, plural)
        
        return UTF8Translations(translations)
    
    def set_language(self, language: str) -> bool:
        """Switch to a different language at runtime.
        
        Args:
            language: Language code (e.g., "en", "zh_CN")
            
        Returns:
            True if language switch was successful
        """
        if language not in self.SUPPORTED_LANGUAGES:
            print(f"Unsupported language: {language}")
            return False
        
        if language == self._current_language:
            return True
        
        if self._load_translation(language):
            self.language_changed.emit(language)
            return True
        
        return False
    
    def get_current_language(self) -> str:
        """Get the current language code."""
        return self._current_language
    
    def get_language_name(self, code: Optional[str] = None) -> str:
        """Get display name for a language code.
        
        Args:
            code: Language code, or None for current language
            
        Returns:
            Human-readable language name
        """
        code = code or self._current_language
        return self.SUPPORTED_LANGUAGES.get(code, code)
    
    def get_available_languages(self) -> List[str]:
        """Get list of available language codes."""
        return list(self.SUPPORTED_LANGUAGES.keys())
    
    def tr(self, message: str, context: Optional[str] = None) -> str:
        """Translate a message string.
        
        Args:
            message: The string to translate
            context: Optional context for disambiguation (not used in basic impl)
            
        Returns:
            Translated string, or original if no translation found
        """
        if not self._translation:
            return message
        
        try:
            # Use gettext's gettext() function
            translated = self._translation.gettext(message)
            return translated if translated else message
        except Exception:
            return message
    
    def tr_n(self, singular: str, plural: str, n: int) -> str:
        """Translate a plural form.
        
        Args:
            singular: Singular form
            plural: Plural form  
            n: Count
            
        Returns:
            Translated string with appropriate plural form
        """
        if not self._translation:
            return singular if n == 1 else plural
        
        try:
            return self._translation.ngettext(singular, plural, n)
        except Exception:
            return singular if n == 1 else plural


# Global i18n manager instance
_i18n_manager: Optional[I18NManager] = None


def init_i18n(locales_dir: Optional[str] = None) -> None:
    """Initialize the global i18n manager.
    
    Must be called once at application startup.
    """
    global _i18n_manager
    _i18n_manager = I18NManager()
    _i18n_manager.init(locales_dir)


def get_i18n() -> I18NManager:
    """Get the global i18n manager instance.
    
    Raises:
        RuntimeError: If i18n not initialized
    """
    if _i18n_manager is None:
        raise RuntimeError("i18n not initialized. Call init_i18n() first.")
    return _i18n_manager


def tr(message: str) -> str:
    """Convenience function for translating strings.
    
    Args:
        message: String to translate
        
    Returns:
        Translated string
    """
    return get_i18n().tr(message)


def tr_n(singular: str, plural: str, n: int) -> str:
    """Convenience function for plural translations.
    
    Args:
        singular: Singular form
        plural: Plural form
        n: Count
        
    Returns:
        Translated string with appropriate plural form
    """
    return get_i18n().tr_n(singular, plural, n)
