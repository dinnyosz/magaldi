-- Magaldi Database Schema
-- This file is executed automatically when the MySQL container starts for the first time.

-- Use the magaldi database (created via MYSQL_DATABASE env var)
USE magaldi;

-- =============================================================================
-- REPOSITORIES
-- =============================================================================

CREATE TABLE repositories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scope VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    tags JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY unique_scope_name (scope, name),
    INDEX idx_scope (scope)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================================================
-- FILE STATES
-- =============================================================================

CREATE TABLE file_states (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scope VARCHAR(100) NOT NULL,
    repository VARCHAR(255) NOT NULL,
    username VARCHAR(100) NOT NULL,
    relative_path VARCHAR(1024) NOT NULL,

    file_hash VARCHAR(64),
    language VARCHAR(50),
    file_size INT UNSIGNED,
    line_count INT UNSIGNED,

    is_deleted BOOLEAN DEFAULT FALSE,

    parsed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,

    UNIQUE KEY unique_file (scope, repository, username, relative_path(255)),
    INDEX idx_scope_repo (scope, repository),
    INDEX idx_username (username),
    INDEX idx_expiry (expires_at),
    INDEX idx_hash (file_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================================================
-- CODE ELEMENTS
-- =============================================================================

CREATE TABLE code_elements (
    element_id VARCHAR(512) PRIMARY KEY,
    scope VARCHAR(100) NOT NULL,
    repository VARCHAR(255) NOT NULL,
    username VARCHAR(100) NOT NULL,
    relative_path VARCHAR(1024) NOT NULL,

    element_type VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    line_start INT UNSIGNED NOT NULL,
    line_end INT UNSIGNED,

    raw_code MEDIUMTEXT,
    signature VARCHAR(1024),
    docstring TEXT,

    level TINYINT UNSIGNED NOT NULL,
    parent_id VARCHAR(512),

    language VARCHAR(50),
    decorators JSON,
    visibility VARCHAR(20),
    is_async BOOLEAN DEFAULT FALSE,

    summary_status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',
    embedding_status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',
    summary TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    expires_at DATETIME,

    INDEX idx_scope_repo (scope, repository),
    INDEX idx_username (username),
    INDEX idx_parent (parent_id),
    INDEX idx_expiry (expires_at),
    INDEX idx_type (element_type),
    INDEX idx_level (level),
    INDEX idx_summary_status (summary_status),
    INDEX idx_embedding_status (embedding_status),
    FOREIGN KEY (parent_id) REFERENCES code_elements(element_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================================================
-- REPOSITORY LANGUAGES
-- =============================================================================

CREATE TABLE repository_languages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scope VARCHAR(100) NOT NULL,
    repository VARCHAR(255) NOT NULL,
    language VARCHAR(50) NOT NULL,
    file_count INT UNSIGNED DEFAULT 0,
    line_count INT UNSIGNED DEFAULT 0,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY unique_repo_lang (scope, repository, language),
    INDEX idx_scope_repo (scope, repository)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================================================
-- SUMMARIZATION JOBS
-- =============================================================================

CREATE TABLE summarization_jobs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    element_id VARCHAR(512) NOT NULL,
    level TINYINT UNSIGNED NOT NULL,
    parent_element_id VARCHAR(512),

    status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',
    dependencies_met BOOLEAN DEFAULT FALSE,
    priority INT DEFAULT 0,

    worker_id VARCHAR(100),
    claimed_at DATETIME,
    completed_at DATETIME,
    retry_count TINYINT UNSIGNED DEFAULT 0,
    error_message TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY unique_element (element_id),
    INDEX idx_status_level (status, level, dependencies_met),
    INDEX idx_worker (worker_id),
    INDEX idx_priority (priority DESC),
    FOREIGN KEY (element_id) REFERENCES code_elements(element_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================================================
-- EMBEDDING JOBS
-- =============================================================================

CREATE TABLE embedding_jobs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    element_id VARCHAR(512) NOT NULL,

    status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',

    worker_id VARCHAR(100),
    claimed_at DATETIME,
    completed_at DATETIME,
    retry_count TINYINT UNSIGNED DEFAULT 0,
    error_message TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY unique_element (element_id),
    INDEX idx_status (status),
    INDEX idx_worker (worker_id),
    FOREIGN KEY (element_id) REFERENCES code_elements(element_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================================================
-- ELEMENT DEPENDENCIES
-- =============================================================================

CREATE TABLE element_dependencies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    source_element_id VARCHAR(512) NOT NULL,
    target_element_id VARCHAR(512) NOT NULL,
    relationship_type ENUM('import', 'call', 'inherit', 'use') NOT NULL,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY unique_dep (source_element_id, target_element_id, relationship_type),
    INDEX idx_source (source_element_id),
    INDEX idx_target (target_element_id),
    FOREIGN KEY (source_element_id) REFERENCES code_elements(element_id) ON DELETE CASCADE,
    FOREIGN KEY (target_element_id) REFERENCES code_elements(element_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================================================
-- GRANTS
-- =============================================================================

-- Ensure magaldi user has full access (user created via MYSQL_USER env var)
GRANT ALL PRIVILEGES ON magaldi.* TO 'magaldi'@'%';
FLUSH PRIVILEGES;
