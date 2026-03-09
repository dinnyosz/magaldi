"""Tests for visibility detection across all supported languages.

Covers:
- PHP: public/private/protected methods, properties, constants
- TypeScript/JavaScript: public/private/protected methods, fields, ES private fields (#field)
- Rust: pub/pub(crate)/pub(super)/private for functions, structs, enums, traits, methods
- Java: public/private/protected/package-private for classes, methods, fields
- Python: name-convention based (_private, __private, public)
"""

from pathlib import Path

import pytest

from magaldi_core.change_detection import FileInfo
from magaldi_core.parsers.base import CodeElement

# =============================================================================
# Helpers
# =============================================================================


def _parse(language: str, code: str, filename: str = "test") -> list[CodeElement]:
    """Parse code with the appropriate parser and return elements."""
    from magaldi_core.code_parser import get_parser

    ext_map = {
        "php": ".php",
        "typescript": ".ts",
        "javascript": ".js",
        "rust": ".rs",
        "java": ".java",
        "python": ".py",
    }
    ext = ext_map.get(language, f".{language}")
    filepath = f"{filename}{ext}"

    parser = get_parser(language)
    file_info = FileInfo(
        relative_path=filepath,
        absolute_path=Path(f"/fake/{filepath}"),
        language=language,
    )
    return parser.parse(code, file_info, "scope", "repo", "main")


def _find(elements: list[CodeElement], name: str) -> CodeElement:
    """Find element by name. Raises if not found."""
    for e in elements:
        if e.name == name:
            return e
    names = [e.name for e in elements]
    raise ValueError(f"Element {name!r} not found. Available: {names}")


# =============================================================================
# PHP Visibility
# =============================================================================


class TestPhpVisibility:
    """Test PHP visibility detection for methods, properties, and constants."""

    PHP_CODE = """<?php
class UserService {
    private $db;
    protected $cache;
    public $name;

    public function findUser($id) { return $this->db->find($id); }
    private function configureDefaults() { $this->cache = []; }
    protected function validate($data) { return true; }
    function noModifier() { return null; }

    const PUBLIC_CONST = 1;
    private const PRIVATE_CONST = 2;
    protected const PROTECTED_CONST = 3;
}
"""

    @pytest.fixture
    def elements(self):
        return _parse("php", self.PHP_CODE)

    def test_public_method(self, elements):
        assert _find(elements, "findUser").visibility == "public"

    def test_private_method(self, elements):
        assert _find(elements, "configureDefaults").visibility == "private"

    def test_protected_method(self, elements):
        assert _find(elements, "validate").visibility == "protected"

    def test_no_modifier_defaults_to_public(self, elements):
        assert _find(elements, "noModifier").visibility == "public"

    def test_private_property(self, elements):
        assert _find(elements, "db").visibility == "private"

    def test_protected_property(self, elements):
        assert _find(elements, "cache").visibility == "protected"

    def test_public_property(self, elements):
        assert _find(elements, "name").visibility == "public"

    def test_public_constant(self, elements):
        assert _find(elements, "PUBLIC_CONST").visibility == "public"

    def test_private_constant(self, elements):
        assert _find(elements, "PRIVATE_CONST").visibility == "private"

    def test_protected_constant(self, elements):
        assert _find(elements, "PROTECTED_CONST").visibility == "protected"


# =============================================================================
# TypeScript Visibility
# =============================================================================


class TestTypescriptVisibility:
    """Test TypeScript visibility detection for methods and fields."""

    TS_CODE = """
class UserService {
    private db: Database;
    protected cache: Map<string, any>;
    public name: string;
    #secretKey: string;

    public findUser(id: string): User {
        return this.db.find(id);
    }

    private configureDefaults(): void {
        this.cache = new Map();
    }

    protected validate(data: any): boolean {
        return true;
    }

    noModifier(): void {}
}
"""

    @pytest.fixture
    def elements(self):
        return _parse("typescript", self.TS_CODE)

    def test_public_method(self, elements):
        assert _find(elements, "findUser").visibility == "public"

    def test_private_method(self, elements):
        assert _find(elements, "configureDefaults").visibility == "private"

    def test_protected_method(self, elements):
        assert _find(elements, "validate").visibility == "protected"

    def test_no_modifier_defaults_to_public(self, elements):
        assert _find(elements, "noModifier").visibility == "public"

    def test_private_field(self, elements):
        assert _find(elements, "db").visibility == "private"

    def test_protected_field(self, elements):
        assert _find(elements, "cache").visibility == "protected"

    def test_public_field(self, elements):
        assert _find(elements, "name").visibility == "public"

    def test_es_private_field(self, elements):
        """ES private fields (#field) should be detected as private."""
        assert _find(elements, "#secretKey").visibility == "private"


# =============================================================================
# Rust Visibility
# =============================================================================


class TestRustVisibility:
    """Test Rust visibility detection for functions, structs, enums, methods."""

    RUST_CODE = """
pub fn public_function() -> i32 { 42 }
fn private_function() -> i32 { 0 }
pub(crate) fn crate_function() -> bool { true }
pub(super) fn super_function() {}

pub struct PublicStruct { pub field: i32, secret: String }
struct PrivateStruct { data: Vec<u8> }
pub(crate) struct CrateStruct {}

pub enum PublicEnum { A, B }
enum PrivateEnum { X, Y }

pub trait PublicTrait { fn required(&self); }
trait PrivateTrait { fn internal(&self); }

impl PublicStruct {
    pub fn new() -> Self { PublicStruct { field: 0, secret: String::new() } }
    fn private_method(&self) -> i32 { self.field }
}

pub const PUBLIC_CONST: i32 = 42;
const PRIVATE_CONST: i32 = 0;

pub type PublicAlias = Vec<String>;
type PrivateAlias = Vec<u8>;
"""

    @pytest.fixture
    def elements(self):
        return _parse("rust", self.RUST_CODE)

    # Functions
    def test_pub_function(self, elements):
        assert _find(elements, "public_function").visibility == "public"

    def test_private_function(self, elements):
        assert _find(elements, "private_function").visibility == "private"

    def test_pub_crate_function(self, elements):
        assert _find(elements, "crate_function").visibility == "pub(crate)"

    def test_pub_super_function(self, elements):
        assert _find(elements, "super_function").visibility == "pub(super)"

    # Structs (treated as class)
    def test_pub_struct(self, elements):
        structs = [e for e in elements if e.name == "PublicStruct" and e.element_type == "class" and "impl" not in e.decorators]
        assert len(structs) == 1
        assert structs[0].visibility == "public"

    def test_private_struct(self, elements):
        assert _find(elements, "PrivateStruct").visibility == "private"

    def test_pub_crate_struct(self, elements):
        assert _find(elements, "CrateStruct").visibility == "pub(crate)"

    # Enums
    def test_pub_enum(self, elements):
        assert _find(elements, "PublicEnum").visibility == "public"

    def test_private_enum(self, elements):
        assert _find(elements, "PrivateEnum").visibility == "private"

    # Traits
    def test_pub_trait(self, elements):
        assert _find(elements, "PublicTrait").visibility == "public"

    def test_private_trait(self, elements):
        assert _find(elements, "PrivateTrait").visibility == "private"

    # Impl methods
    def test_pub_method_in_impl(self, elements):
        assert _find(elements, "new").visibility == "public"

    def test_private_method_in_impl(self, elements):
        assert _find(elements, "private_method").visibility == "private"

    # Constants
    def test_pub_constant(self, elements):
        assert _find(elements, "PUBLIC_CONST").visibility == "public"

    def test_private_constant(self, elements):
        assert _find(elements, "PRIVATE_CONST").visibility == "private"

    # Type aliases
    def test_pub_type_alias(self, elements):
        assert _find(elements, "PublicAlias").visibility == "public"

    def test_private_type_alias(self, elements):
        assert _find(elements, "PrivateAlias").visibility == "private"


# =============================================================================
# Java Visibility
# =============================================================================


class TestJavaVisibility:
    """Test Java visibility detection for classes, methods, and fields."""

    JAVA_CODE = """
public class UserService {
    private Database db;
    protected Cache cache;
    public String name;
    String packageField;

    public User findUser(String id) { return db.find(id); }
    private void configureDefaults() { this.cache = new Cache(); }
    protected boolean validate(Object data) { return true; }
    void packagePrivate() {}

    private static class Inner {}
}
"""

    @pytest.fixture
    def elements(self):
        return _parse("java", self.JAVA_CODE)

    # Class
    def test_public_class(self, elements):
        assert _find(elements, "UserService").visibility == "public"

    def test_private_nested_class(self, elements):
        assert _find(elements, "Inner").visibility == "private"

    # Methods
    def test_public_method(self, elements):
        assert _find(elements, "findUser").visibility == "public"

    def test_private_method(self, elements):
        assert _find(elements, "configureDefaults").visibility == "private"

    def test_protected_method(self, elements):
        assert _find(elements, "validate").visibility == "protected"

    def test_package_private_method(self, elements):
        """No modifier in Java = package-private, not public."""
        assert _find(elements, "packagePrivate").visibility == "package"

    # Fields
    def test_private_field(self, elements):
        assert _find(elements, "db").visibility == "private"

    def test_protected_field(self, elements):
        assert _find(elements, "cache").visibility == "protected"

    def test_public_field(self, elements):
        assert _find(elements, "name").visibility == "public"

    def test_package_private_field(self, elements):
        assert _find(elements, "packageField").visibility == "package"


# =============================================================================
# Python Visibility (existing behavior - name-convention based)
# =============================================================================


class TestPythonVisibility:
    """Test Python visibility detection (name-convention based)."""

    PY_CODE = """
class MyClass:
    def public_method(self):
        pass

    def _protected_method(self):
        pass

    def __private_method(self):
        pass

    def __dunder__(self):
        pass
"""

    @pytest.fixture
    def elements(self):
        return _parse("python", self.PY_CODE)

    def test_public_method(self, elements):
        assert _find(elements, "public_method").visibility == "public"

    def test_protected_method(self, elements):
        assert _find(elements, "_protected_method").visibility == "protected"

    def test_private_method(self, elements):
        assert _find(elements, "__private_method").visibility == "private"

    def test_dunder_method_is_public(self, elements):
        """Dunder methods like __init__ should be public."""
        assert _find(elements, "__dunder__").visibility == "public"


# =============================================================================
# Cross-language: detect_public_api integration
# =============================================================================


class TestPublicApiDetection:
    """Verify detect_public_api correctly uses the new visibility values."""

    def test_rust_pub_crate_not_public_api(self):
        from magaldi_core.extractors.detection import detect_public_api

        assert detect_public_api("my_func", [], "pub(crate)", "rust") is False

    def test_rust_pub_super_not_public_api(self):
        from magaldi_core.extractors.detection import detect_public_api

        assert detect_public_api("my_func", [], "pub(super)", "rust") is False

    def test_java_package_not_public_api(self):
        from magaldi_core.extractors.detection import detect_public_api

        assert detect_public_api("myMethod", [], "package", "java") is False

    def test_public_is_public_api(self):
        from magaldi_core.extractors.detection import detect_public_api

        assert detect_public_api("myMethod", [], "public", "java") is True

    def test_private_not_public_api(self):
        from magaldi_core.extractors.detection import detect_public_api

        assert detect_public_api("myMethod", [], "private", "java") is False
