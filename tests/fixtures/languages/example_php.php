<?php

/**
 * Example PHP file exercising all extractable element types.
 *
 * This fixture ensures the magaldi index always contains PHP-specific
 * elements (interface, trait, enum, attributes, visibility) when parsing its own repo.
 */

namespace App\Example;

use App\Contracts\Cacheable;
use App\Events\EventDispatcher;
use Psr\Log\LoggerInterface;
use RuntimeException;

// Constants
const MAX_CONNECTIONS = 10;
const DEFAULT_TIMEOUT = 30;
const SUPPORTED_DRIVERS = ['mysql', 'pgsql', 'sqlite'];

// --- Enums ---

enum Status: string
{
    case PENDING = 'pending';
    case ACTIVE = 'active';
    case SUSPENDED = 'suspended';
    case DELETED = 'deleted';

    public function isActive(): bool
    {
        return $this === self::ACTIVE;
    }

    public function label(): string
    {
        return match($this) {
            self::PENDING => 'Pending Review',
            self::ACTIVE => 'Active',
            self::SUSPENDED => 'Suspended',
            self::DELETED => 'Deleted',
        };
    }
}

enum Priority: int
{
    case LOW = 1;
    case MEDIUM = 5;
    case HIGH = 10;
    case CRITICAL = 100;
}

// --- Interfaces ---

interface Repository
{
    public function find(string $id): ?object;
    public function findAll(array $criteria = []): array;
    public function save(object $entity): void;
    public function delete(string $id): bool;
}

interface Transformer
{
    public function transform(mixed $data): array;
    public function supportsType(string $type): bool;
}

interface EventSubscriber
{
    public static function getSubscribedEvents(): array;
}

// --- Traits ---

trait HasTimestamps
{
    private ?\DateTimeInterface $createdAt = null;
    private ?\DateTimeInterface $updatedAt = null;

    public function getCreatedAt(): ?\DateTimeInterface
    {
        return $this->createdAt;
    }

    public function setCreatedAt(\DateTimeInterface $date): void
    {
        $this->createdAt = $date;
    }

    public function touch(): void
    {
        $this->updatedAt = new \DateTimeImmutable();
    }
}

trait HasSlug
{
    private string $slug = '';

    public function getSlug(): string
    {
        return $this->slug;
    }

    public function generateSlug(string $value): string
    {
        $this->slug = strtolower(trim(preg_replace('/[^A-Za-z0-9-]+/', '-', $value), '-'));
        return $this->slug;
    }
}

trait Loggable
{
    private ?LoggerInterface $logger = null;

    public function setLogger(LoggerInterface $logger): void
    {
        $this->logger = $logger;
    }

    protected function log(string $level, string $message, array $context = []): void
    {
        $this->logger?->log($level, $message, $context);
    }
}

// --- Classes ---

#[Attribute]
class Route
{
    public function __construct(
        public readonly string $path,
        public readonly string $method = 'GET',
    ) {}
}

#[Attribute]
class Validate
{
    public function __construct(
        public readonly string $rule,
    ) {}
}

abstract class BaseEntity
{
    use HasTimestamps;

    public function __construct(
        protected readonly string $id,
    ) {
        $this->setCreatedAt(new \DateTimeImmutable());
    }

    public function getId(): string
    {
        return $this->id;
    }

    abstract public function toArray(): array;
}

class User extends BaseEntity implements EventSubscriber
{
    use HasSlug;
    use Loggable;

    private Status $status = Status::PENDING;

    public function __construct(
        string $id,
        private string $name,
        #[Validate('email')]
        private string $email,
        private Priority $priority = Priority::MEDIUM,
    ) {
        parent::__construct($id);
        $this->generateSlug($name);
    }

    public function activate(): void
    {
        $this->status = Status::ACTIVE;
        $this->touch();
        $this->log('info', 'User activated', ['id' => $this->id]);
    }

    public function suspend(): void
    {
        $this->status = Status::SUSPENDED;
        $this->touch();
    }

    public function getStatus(): Status
    {
        return $this->status;
    }

    public function toArray(): array
    {
        return [
            'id' => $this->id,
            'name' => $this->name,
            'email' => $this->email,
            'status' => $this->status->value,
            'slug' => $this->getSlug(),
        ];
    }

    public static function getSubscribedEvents(): array
    {
        return ['user.created' => 'onUserCreated'];
    }
}

final readonly class UserDTO
{
    public function __construct(
        public string $id,
        public string $name,
        public string $email,
        public string $status,
    ) {}

    public static function fromEntity(User $user): self
    {
        $data = $user->toArray();
        return new self(
            id: $data['id'],
            name: $data['name'],
            email: $data['email'],
            status: $data['status'],
        );
    }
}

class UserRepository implements Repository
{
    private array $store = [];

    public function find(string $id): ?User
    {
        return $this->store[$id] ?? null;
    }

    public function findAll(array $criteria = []): array
    {
        return array_values($this->store);
    }

    public function save(object $entity): void
    {
        if (!$entity instanceof User) {
            throw new RuntimeException('Expected User entity');
        }
        $this->store[$entity->getId()] = $entity;
    }

    public function delete(string $id): bool
    {
        if (isset($this->store[$id])) {
            unset($this->store[$id]);
            return true;
        }
        return false;
    }
}

// --- Functions ---

function createUser(string $name, string $email): User
{
    $id = bin2hex(random_bytes(16));
    return new User($id, $name, $email);
}

function validateEmail(string $email): bool
{
    return filter_var($email, FILTER_VALIDATE_EMAIL) !== false;
}

#[Route('/api/users', method: 'GET')]
function listUsers(UserRepository $repo): array
{
    return array_map(
        fn(User $user) => UserDTO::fromEntity($user),
        $repo->findAll()
    );
}
