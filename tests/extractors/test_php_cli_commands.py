"""Tests for PHP CLI command extraction (Symfony Console, Laravel Artisan)."""

import pytest
from tree_sitter import Language, Parser

import tree_sitter_php as ts_php

from magaldi_core.extractors.patterns.php.cli_commands import (
    extract_php_cli_commands,
)


class TestSymfonyConsoleCommands:
    """Test extraction of Symfony Console commands."""

    @pytest.fixture
    def php_parser(self):
        """Create a PHP parser."""
        php_lang = Language(ts_php.language_php())
        return Parser(php_lang)

    def test_as_command_attribute_basic(self, php_parser):
        """Test extraction of basic #[AsCommand] attribute."""
        code = """<?php
namespace App\\Command;

use Symfony\\Component\\Console\\Attribute\\AsCommand;
use Symfony\\Component\\Console\\Command\\Command;

#[AsCommand(name: 'app:create-user')]
class CreateUserCommand extends Command
{
    public function __invoke(): int
    {
        return Command::SUCCESS;
    }
}
"""
        tree = php_parser.parse(code.encode())
        commands = extract_php_cli_commands(tree, code.split("\n"))

        assert len(commands) == 1
        assert commands[0].name == "app:create-user"
        assert commands[0].framework == "symfony-console"

    def test_as_command_with_description(self, php_parser):
        """Test extraction of #[AsCommand] with description."""
        code = """<?php
#[AsCommand(
    name: 'mail:send',
    description: 'Send marketing emails to users'
)]
class SendEmailsCommand
{
}
"""
        tree = php_parser.parse(code.encode())
        commands = extract_php_cli_commands(tree, code.split("\n"))

        assert len(commands) == 1
        assert commands[0].name == "mail:send"

    def test_as_command_positional_arg(self, php_parser):
        """Test extraction of #[AsCommand] with positional argument."""
        code = """<?php
#[AsCommand('hello:world')]
class HelloWorldCommand
{
}
"""
        tree = php_parser.parse(code.encode())
        commands = extract_php_cli_commands(tree, code.split("\n"))

        assert len(commands) == 1
        assert commands[0].name == "hello:world"


class TestLaravelArtisanCommands:
    """Test extraction of Laravel Artisan commands."""

    @pytest.fixture
    def php_parser(self):
        """Create a PHP parser."""
        php_lang = Language(ts_php.language_php())
        return Parser(php_lang)

    def test_artisan_signature_basic(self, php_parser):
        """Test extraction of basic Laravel Artisan signature."""
        code = """<?php
namespace App\\Console\\Commands;

use Illuminate\\Console\\Command;

class SendEmails extends Command
{
    protected $signature = 'mail:send {user}';
    protected $description = 'Send a marketing email to a user';

    public function handle(): void
    {
    }
}
"""
        tree = php_parser.parse(code.encode())
        commands = extract_php_cli_commands(tree, code.split("\n"))

        assert len(commands) == 1
        assert commands[0].name == "mail:send"
        assert commands[0].framework == "laravel-artisan"
        assert len(commands[0].options) == 1
        assert commands[0].options[0]["name"] == "user"
        assert commands[0].options[0]["type"] == "argument"

    def test_artisan_signature_with_options(self, php_parser):
        """Test extraction of Laravel Artisan signature with options."""
        code = """<?php
class ProcessData extends Command
{
    protected $signature = 'data:process {file} {--queue=default} {--force}';

    public function handle(): void
    {
    }
}
"""
        tree = php_parser.parse(code.encode())
        commands = extract_php_cli_commands(tree, code.split("\n"))

        assert len(commands) == 1
        assert commands[0].name == "data:process"
        # Should have 1 argument and 2 options
        args = [o for o in commands[0].options if o["type"] == "argument"]
        opts = [o for o in commands[0].options if o["type"] == "option"]
        assert len(args) == 1
        assert args[0]["name"] == "file"
        assert len(opts) == 2

    def test_artisan_signature_optional_arg(self, php_parser):
        """Test extraction of Laravel Artisan signature with optional argument."""
        code = """<?php
class GreetCommand extends Command
{
    protected $signature = 'greet {name?}';
}
"""
        tree = php_parser.parse(code.encode())
        commands = extract_php_cli_commands(tree, code.split("\n"))

        assert len(commands) == 1
        assert commands[0].name == "greet"
        assert len(commands[0].options) == 1
        assert commands[0].options[0]["name"] == "name"
        assert commands[0].options[0]["required"] is False


class TestNoCliCommands:
    """Test that non-CLI classes are not detected."""

    @pytest.fixture
    def php_parser(self):
        """Create a PHP parser."""
        php_lang = Language(ts_php.language_php())
        return Parser(php_lang)

    def test_regular_class_not_detected(self, php_parser):
        """Test that regular classes without CLI attributes are not detected."""
        code = """<?php
class UserController
{
    public function index()
    {
        return [];
    }
}
"""
        tree = php_parser.parse(code.encode())
        commands = extract_php_cli_commands(tree, code.split("\n"))

        assert len(commands) == 0

    def test_route_attribute_not_detected_as_cli(self, php_parser):
        """Test that #[Route] attribute is not detected as CLI command."""
        code = """<?php
#[Route('/users')]
class UserController
{
}
"""
        tree = php_parser.parse(code.encode())
        commands = extract_php_cli_commands(tree, code.split("\n"))

        assert len(commands) == 0
