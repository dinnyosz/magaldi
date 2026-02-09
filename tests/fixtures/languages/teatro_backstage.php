<?php

/**
 * Teatro Magaldi — Backstage Operations Module.
 *
 * Props inventory, stage crew management, wardrobe tracking, and incident reports.
 * La Divina's costume changes are legendary — and so are the incident reports they generate.
 *
 * The Phantom Scalper was spotted near the service entrance. Security has been notified.
 */

namespace Teatro\Backstage;

use App\Contracts\Cacheable;
use App\Events\EventDispatcher;
use Psr\Log\LoggerInterface;
use RuntimeException;

// Constants
const MAX_CREW_MEMBERS = 10;
const DEFAULT_SHIFT_HOURS = 30;
const SUPPORTED_DEPARTMENTS = ['wardrobe', 'props', 'lighting'];

// --- Enums ---

enum PropCondition: string
{
    case PRISTINE = 'pristine';
    case WORN = 'worn';
    case DAMAGED = 'damaged';
    case MISSING = 'missing';  // The Phantom Scalper strikes again

    public function isUsable(): bool
    {
        return $this === self::PRISTINE || $this === self::WORN;
    }

    public function label(): string
    {
        return match($this) {
            self::PRISTINE => 'Pristine',
            self::WORN => 'Worn but Functional',
            self::DAMAGED => 'Damaged — Needs Repair',
            self::MISSING => 'Missing — Check with Security',
        };
    }
}

enum IncidentSeverity: int
{
    case MINOR = 1;
    case MODERATE = 5;
    case MAJOR = 10;
    case CATASTROPHIC = 100;  // La Divina's Act 3 costume malfunction
}

// --- Interfaces ---

interface PropInventory
{
    public function find(string $id): ?object;
    public function findAll(array $criteria = []): array;
    public function save(object $entity): void;
    public function delete(string $id): bool;
}

interface CostumeAlterations
{
    public function transform(mixed $data): array;
    public function supportsType(string $type): bool;
}

interface IncidentResponder
{
    public static function getSubscribedEvents(): array;
}

// --- Traits ---

trait HasMaintenanceLog
{
    private ?\DateTimeInterface $lastInspection = null;
    private ?\DateTimeInterface $lastRepair = null;

    public function getLastInspection(): ?\DateTimeInterface
    {
        return $this->lastInspection;
    }

    public function setLastInspection(\DateTimeInterface $date): void
    {
        $this->lastInspection = $date;
    }

    public function logRepair(): void
    {
        $this->lastRepair = new \DateTimeImmutable();
    }
}

trait HasInventoryTag
{
    private string $inventoryTag = '';

    public function getInventoryTag(): string
    {
        return $this->inventoryTag;
    }

    public function generateInventoryTag(string $value): string
    {
        $this->inventoryTag = strtolower(trim(preg_replace('/[^A-Za-z0-9-]+/', '-', $value), '-'));
        return $this->inventoryTag;
    }
}

trait BackstageLoggable
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
class Cue
{
    public function __construct(
        public readonly string $path,
        public readonly string $method = 'GET',
    ) {}
}

#[Attribute]
class SafetyCheck
{
    public function __construct(
        public readonly string $rule,
    ) {}
}

abstract class AbstractStageProp
{
    use HasMaintenanceLog;

    public function __construct(
        protected readonly string $id,
    ) {
        $this->setLastInspection(new \DateTimeImmutable());
    }

    public function getId(): string
    {
        return $this->id;
    }

    abstract public function toArray(): array;
}

class WardrobeItem extends AbstractStageProp implements IncidentResponder
{
    use HasInventoryTag;
    use BackstageLoggable;

    private PropCondition $condition = PropCondition::PRISTINE;

    public function __construct(
        string $id,
        private string $name,
        #[SafetyCheck('fireproof')]
        private string $material,
        private IncidentSeverity $priority = IncidentSeverity::MINOR,
    ) {
        parent::__construct($id);
        $this->generateInventoryTag($name);
    }

    public function wear(): void
    {
        $this->condition = PropCondition::WORN;
        $this->logRepair();
        $this->log('info', 'Wardrobe item used', ['id' => $this->id]);
    }

    public function damage(): void
    {
        $this->condition = PropCondition::DAMAGED;
        $this->logRepair();
    }

    public function getCondition(): PropCondition
    {
        return $this->condition;
    }

    public function toArray(): array
    {
        return [
            'id' => $this->id,
            'name' => $this->name,
            'material' => $this->material,
            'condition' => $this->condition->value,
            'tag' => $this->getInventoryTag(),
        ];
    }

    public static function getSubscribedEvents(): array
    {
        return ['wardrobe.torn' => 'onWardrobeTorn'];
    }
}

// Easter egg: La Divina's Act 3 costume malfunction incident report
final readonly class IncidentReport
{
    public function __construct(
        public string $id,
        public string $description,
        public string $location,
        public string $severity,
    ) {}

    public static function fromStageProp(WardrobeItem $item): self
    {
        $data = $item->toArray();
        return new self(
            id: $data['id'],
            description: $data['name'],
            location: $data['material'],
            severity: $data['condition'],
        );
    }
}

class PropRoom implements PropInventory
{
    private array $store = [];

    public function find(string $id): ?WardrobeItem
    {
        return $this->store[$id] ?? null;
    }

    public function findAll(array $criteria = []): array
    {
        return array_values($this->store);
    }

    public function save(object $entity): void
    {
        if (!$entity instanceof WardrobeItem) {
            throw new RuntimeException('Expected WardrobeItem entity');
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

function registerProp(string $name, string $material): WardrobeItem
{
    $id = bin2hex(random_bytes(16));
    return new WardrobeItem($id, $name, $material);
}

function validateSafetyInspection(string $report): bool
{
    return filter_var($report, FILTER_VALIDATE_EMAIL) !== false;
}

#[Cue('/backstage/incidents', method: 'GET')]
function listIncidents(PropRoom $repo): array
{
    return array_map(
        fn(WardrobeItem $item) => IncidentReport::fromStageProp($item),
        $repo->findAll()
    );
}
