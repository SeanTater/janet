# ChunkHound Architecture Reference
## Comprehensive Guide for Agentic Code Development Inspiration

### Overview

ChunkHound is a sophisticated **Model Context Protocol (MCP) Server** that provides semantic and regex search capabilities for codebases. It serves as an excellent architectural reference for building **agentic code development tools** due to its robust design patterns, real-time processing capabilities, and clean separation of concerns.

### Key Architectural Strengths

- **MCP-First Design**: Built specifically for AI assistant integration
- **Real-Time File Monitoring**: Sophisticated file watching with debouncing and priority handling
- **Multi-Modal Search**: Combines semantic embeddings, regex patterns, and fuzzy matching
- **Pluggable Architecture**: Clean abstractions for databases, embeddings, and parsers
- **Production-Ready**: Comprehensive error handling, logging, and performance optimization

---

## 1. High-Level Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Client (AI Assistant)                    │
├─────────────────────────────────────────────────────────────────┤
│                    JSON-RPC Protocol (stdio)                    │
├─────────────────────────────────────────────────────────────────┤
│                      ChunkHound MCP Server                      │
│  ┌───────────────┬──────────────────┬─────────────────────────┐ │
│  │ Tool Handlers │  Search Services │    Task Coordinator     │ │
│  └───────────────┴──────────────────┴─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Core Service Layer                           │
│  ┌──────────────┬─────────────────┬─────────────────────────┐  │
│  │   Indexing   │   File Watcher  │    Embedding Service    │  │
│  │ Coordinator  │    Manager      │                         │  │
│  └──────────────┴─────────────────┴─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Provider Abstraction                         │
│  ┌──────────────┬─────────────────┬─────────────────────────┐  │
│  │   Database   │    Embedding    │     Language Parser     │  │
│  │   Provider   │    Provider     │      Provider           │  │
│  └──────────────┴─────────────────┴─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                   Storage & Computation                         │
│  ┌──────────────┬─────────────────┬─────────────────────────┐  │
│  │   DuckDB/    │   OpenAI/BGE    │     Tree-sitter         │  │
│  │   LanceDB    │   Embeddings    │     Parsers             │  │
│  └──────────────┴─────────────────┴─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Python 3.10+** with asyncio for concurrency
- **Tree-sitter** for language-aware code parsing
- **DuckDB/LanceDB** for vector storage and search
- **OpenAI/BGE** for embedding generation
- **MCP Protocol** for AI assistant integration
- **Watchdog** for real-time file monitoring

---

## 2. MCP Server Architecture

### JSON-RPC Integration

The MCP server is designed with **clean JSON-RPC communication** as a first-class concern:

```python
# Key Design Patterns
- Logging suppression to prevent JSON-RPC interference
- Structured error responses with helpful diagnostics
- Tool-based architecture matching MCP specifications
- Graceful degradation when services are unavailable
```

### Tool Implementation Pattern

**Search Tools**:
- `search_semantic`: Vector similarity search with OpenAI embeddings
- `search_regex`: Pattern matching with pagination
- `search_fuzzy`: Approximate text matching
- `get_stats`: Database statistics and health metrics
- `health_check`: System status validation

**Tool Handler Architecture**:
```python
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    # 1. Validate arguments and apply constraints
    # 2. Queue operation through TaskCoordinator 
    # 3. Execute with proper error handling
    # 4. Apply response size limiting
    # 5. Return structured JSON response
```

### Server Lifecycle Management

**Initialization Phase**:
1. **Configuration Loading**: Unified config system with environment variable support
2. **Process Detection**: Prevent multiple server instances per database
3. **Database Connection**: Thread-safe initialization with connection pooling
4. **Service Startup**: Coordinated startup of all background services
5. **File Watching**: Real-time monitoring with offline catch-up

**Runtime Phase**:
- **Request Processing**: Priority-based task coordination
- **Background Operations**: Periodic indexing and maintenance
- **Error Recovery**: Graceful handling of transient failures

**Shutdown Phase**:
- **Task Completion**: Graceful drain of pending operations
- **Resource Cleanup**: Database connections, file watchers, temporary files
- **State Persistence**: Checkpoint database and cleanup coordination files

---

## 3. Database Architecture

### Multi-Provider Design

**Provider Abstraction**:
```python
class DatabaseProvider(Protocol):
    def connect(self) -> None: ...
    def search_semantic(self, query_vector, **kwargs): ...
    def search_regex(self, pattern: str, **kwargs): ...
    def create_file(self, file: File) -> FileId: ...
    def create_chunks(self, chunks: list[Chunk]) -> list[ChunkId]: ...
    # 50+ additional methods for complete database operations
```

**Implemented Providers**:
- **DuckDBProvider**: HNSW vector indexes, WAL corruption handling
- **LanceDBProvider**: Columnar storage, native vector operations

### Schema Design

**Multi-Dimensional Vector Storage**:
```sql
-- Separate tables per embedding dimension for optimal performance
CREATE TABLE embeddings_1536 (
    id INTEGER PRIMARY KEY,
    chunk_id INTEGER REFERENCES chunks(id),
    provider TEXT NOT NULL,          -- "openai", "bge", etc.
    model TEXT NOT NULL,             -- "text-embedding-3-small"
    embedding FLOAT[1536],           -- Fixed-size vector array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- HNSW index for cosine similarity search
CREATE INDEX idx_hnsw_1536 ON embeddings_1536 
USING HNSW (embedding) WITH (similarity = cosine);
```

**Core Entity Storage**:
```sql
-- Files with metadata
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    language TEXT,                   -- Programming language
    size INTEGER,                    -- File size in bytes
    modified_time TIMESTAMP,         -- Last modification time
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Semantic code chunks
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER REFERENCES files(id),
    chunk_type TEXT NOT NULL,        -- function, class, method, etc.
    symbol TEXT,                     -- Function/class name
    code TEXT NOT NULL,              -- Raw code content
    start_line INTEGER,              -- 1-based line numbers
    end_line INTEGER,
    language TEXT
);
```

### Transaction & Consistency Patterns

**Serial Execution Pattern**:
```python
# All database operations serialized to prevent race conditions
class SerialDatabaseProvider:
    def __init__(self, provider: DatabaseProvider):
        self._provider = provider
        self._executor = SerialExecutor()
    
    async def execute(self, operation: str, *args):
        return await self._executor.execute(operation, *args)
```

**ACID Guarantees**:
- **Atomic Operations**: Bulk inserts in single transactions
- **Consistency**: Foreign key constraints with cascading deletes
- **Isolation**: Thread-safe operation serialization
- **Durability**: WAL with periodic checkpointing

---

## 4. Indexing & Search Architecture

### Language-Aware Parsing

**Tree-sitter Integration**:
```python
# Parser abstraction for 20+ languages
class LanguageParser(Protocol):
    def parse_file(self, file_path: Path) -> ParseResult: ...
    def extract_symbols(self, tree: Tree) -> list[Symbol]: ...
    def get_chunk_boundaries(self, content: str) -> list[Range]: ...

# Supported languages with specialized parsers
SUPPORTED_LANGUAGES = {
    Language.PYTHON: PythonParser(),
    Language.TYPESCRIPT: TypeScriptParser(), 
    Language.JAVA: JavaParser(),
    Language.CSHARP: CSharpParser(),
    # ... additional language support
}
```

**Chunk Type Taxonomy**:
```python
class ChunkType(Enum):
    FUNCTION = "function"
    CLASS = "class" 
    METHOD = "method"
    INTERFACE = "interface"
    ENUM = "enum"
    VARIABLE = "variable"
    IMPORT = "import"
    COMMENT = "comment"
    # ... 20+ semantic chunk types
```

### Search Service Architecture

**Multi-Modal Search**:
```python
class SearchService:
    async def search_semantic(
        self, 
        query_vector: list[float],
        provider: str,
        model: str,
        threshold: float = None,
        path_filter: str = None
    ) -> tuple[list[SearchResult], PaginationInfo]: ...
    
    async def search_regex(
        self,
        pattern: str,
        path_filter: str = None
    ) -> tuple[list[SearchResult], PaginationInfo]: ...
```

**Embedding Management**:
```python
class EmbeddingService:
    async def generate_embeddings(
        self, 
        texts: list[str], 
        provider: str = "openai"
    ) -> EmbeddingResult: ...
    
    def register_provider(self, provider: EmbeddingProvider): ...
    def list_providers(self) -> list[str]: ...
```

---

## 5. Real-Time File Monitoring

### Multi-Layered Watching System

**File System Monitoring**:
```python
class FileWatcherManager:
    def __init__(self):
        self._watchers: dict[Path, FileWatcher] = {}
        self._event_handler = ChunkHoundEventHandler()
        
    async def initialize(self, callback: Callable[[Path, str], None]):
        # 1. Setup watchdog observers for each watch path
        # 2. Configure event filtering and buffering
        # 3. Start offline catch-up scan
        # 4. Begin real-time monitoring
```

**Event Processing Pipeline**:
```
File Change → Event Buffer → Priority Queue → Database Update
     ↓              ↓             ↓              ↓
  Watchdog    Thread-Safe     Task           Atomic
  Detection    Deque          Coordinator    Transaction
```

### Priority-Based Task Coordination

**Task Priority System**:
```python
class TaskPriority(IntEnum):
    HIGH = 1        # Search operations (user-facing)
    MEDIUM = 5      # Health checks, stats
    LOW = 10        # File updates, embeddings
    BACKGROUND = 20 # Periodic maintenance

class TaskCoordinator:
    async def queue_task(
        self, 
        priority: TaskPriority, 
        task: Callable
    ) -> Any:
        # Ensures search operations are never blocked by file processing
```

**Offline Recovery**:
```python
async def scan_for_offline_changes(self):
    """Process files modified while server was offline"""
    # 1. Compare filesystem timestamps with database records
    # 2. Queue missing/modified files for processing
    # 3. Handle deletions and moves
    # 4. Update database with changes
```

---

## 6. Configuration & Environment Management

### Unified Configuration System

**Hierarchical Configuration**:
```python
class Config:
    def __init__(self, target_dir: Path = None):
        # 1. Load from environment variables
        # 2. Read from .chunkhound.json
        # 3. Apply command-specific defaults
        # 4. Validate configuration consistency

# Configuration validation per command
def validate_for_command(self, command: str) -> list[str]:
    if command == "mcp":
        # MCP servers can run without embedding providers
        return []
    elif command == "search":
        # Search requires embedding configuration
        return self._validate_embedding_config()
```

**Environment Variable Patterns**:
```bash
# Database configuration
CHUNKHOUND_DATABASE__PATH="/path/to/chunkhound.db"
CHUNKHOUND_DATABASE__PROVIDER="duckdb"

# Embedding configuration  
CHUNKHOUND_EMBEDDING__PROVIDER="openai"
CHUNKHOUND_EMBEDDING__API_KEY="sk-..."
CHUNKHOUND_EMBEDDING__MODEL="text-embedding-3-small"

# File watching configuration
CHUNKHOUND_WATCH_PATHS="/project/src:/project/tests"
CHUNKHOUND_WATCH_ENABLED="true"
```

### Project Detection

**Automatic Project Root Discovery**:
```python
def find_project_root(start_path: Path = None) -> Path:
    """Find project root by looking for indicators"""
    indicators = [".git", "pyproject.toml", "package.json", 
                  "Cargo.toml", "go.mod", ".chunkhound.json"]
    
    # Walk up directory tree until indicator found
    current = start_path or Path.cwd()
    while current != current.parent:
        if any((current / indicator).exists() for indicator in indicators):
            return current
        current = current.parent
    
    return start_path or Path.cwd()
```

---

## 7. Error Handling & Resilience

### MCP-Safe Error Management

**JSON-RPC Compatibility**:
```python
# Critical: Suppress all logging to prevent JSON-RPC interference
logging.disable(logging.CRITICAL)
for logger_name in ["", "mcp", "server", "fastmcp"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

def _log_processing_error(e: Exception, event_type: str, file_path: Path):
    """Log errors without corrupting JSON-RPC"""
    debug_mode = os.getenv("CHUNKHOUND_DEBUG", "").lower() in ("true", "1")
    if debug_mode:
        print(f"[MCP-SERVER] Error: {e}", file=sys.stderr)
```

**Structured Error Responses**:
```python
def send_error_response(message_id: Any, code: int, message: str, data: dict = None):
    """Send JSON-RPC error response with helpful diagnostics"""
    error_response = {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {
            "code": code,
            "message": message,
            "data": {
                "details": "Human-readable explanation",
                "suggestion": "How to fix the issue",
                "help": ["Step 1", "Step 2", "Step 3"]
            }
        }
    }
```

### Graceful Degradation Patterns

**Service Availability Handling**:
```python
# Example: Embedding service optional for MCP servers
try:
    provider = EmbeddingProviderFactory.create_provider(config.embedding)
    embedding_manager.register_provider(provider)
except ValueError as e:
    # Continue without embeddings - regex search still available
    logger.warning(f"Embedding provider unavailable: {e}")

# Example: File watching graceful failure
if not WATCHDOG_AVAILABLE:
    logger.warning("File watching disabled - install watchdog package")
    # Server continues with manual indexing only
```

**Database Recovery**:
```python
def _handle_wal_corruption(self):
    """Recover from WAL corruption by rebuilding database"""
    try:
        # 1. Backup existing data
        # 2. Create new database file
        # 3. Restore data from backup
        # 4. Rebuild indexes
        logger.info("Database recovered from WAL corruption")
    except Exception as e:
        logger.error(f"Database recovery failed: {e}")
        raise
```

---

## 8. Performance Optimization Strategies

### Bulk Operation Patterns

**Batch Processing**:
```python
async def process_files_batch(self, files: list[Path], batch_size: int = 10):
    """Process files in batches to optimize database operations"""
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        
        # Yield control every 50 files for responsiveness
        if i % 50 == 0:
            await asyncio.sleep(0)
        
        await self._process_batch(batch)
```

**Index Management**:
```python
async def bulk_update_with_index_optimization(self, chunks: list[Chunk]):
    """Optimize bulk operations by temporarily dropping indexes"""
    try:
        # 1. Drop expensive HNSW indexes
        await self._drop_vector_indexes()
        
        # 2. Perform bulk operations
        await self._insert_chunks_bulk(chunks)
        
        # 3. Recreate indexes
        await self._create_vector_indexes()
    except Exception:
        # Rollback and ensure indexes exist
        await self._ensure_indexes_exist()
        raise
```

### Memory Management

**Connection Pooling**:
```python
class ConnectionManager:
    def __init__(self, max_connections: int = 10):
        self._pool: asyncio.Queue[Connection] = asyncio.Queue(max_connections)
        self._active_connections: set[Connection] = set()
    
    async def get_connection(self) -> Connection:
        try:
            return self._pool.get_nowait()
        except asyncio.QueueEmpty:
            return await self._create_connection()
```

**Resource Cleanup**:
```python
async def cleanup(self):
    """Comprehensive resource cleanup"""
    # 1. Stop background tasks
    if self._periodic_indexer:
        await self._periodic_indexer.stop()
    
    # 2. Drain task queues
    if self._task_coordinator:
        await self._task_coordinator.stop()
    
    # 3. Close database connections
    if self._database:
        self._database.disconnect()
    
    # 4. Cleanup temporary files
    self._cleanup_temp_files()
```

---

## 9. Testing & Development Patterns

### Test Architecture

**Integration Testing**:
```python
@pytest.fixture
async def test_database():
    """Create isolated test database"""
    db_path = Path(tempfile.mkdtemp()) / "test.db"
    config = DatabaseConfig(path=str(db_path), provider="duckdb")
    
    database = create_database_with_dependencies(db_path, config)
    yield database
    
    # Cleanup
    database.disconnect()
    shutil.rmtree(db_path.parent)
```

**MCP Server Testing**:
```python
async def test_mcp_server_initialization():
    """Test MCP server startup and tool registration"""
    async with server_lifespan(server) as context:
        # Verify all services initialized
        assert context["db"] is not None
        assert context["embeddings"] is not None
        
        # Test tool availability
        tools = await list_tools()
        tool_names = {tool.name for tool in tools}
        assert "search_semantic" in tool_names
        assert "search_regex" in tool_names
```

### Development Tooling

**Debug Utilities**:
```python
def debug_log(event_type: str, **data):
    """MCP-safe debug logging to file"""
    if os.environ.get("CHUNKHOUND_DEBUG_MODE") == "1":
        debug_file = Path(".mem/debug/chunkhound-debug.jsonl")
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        
        entry = {
            "timestamp": time.time(),
            "event": event_type,
            "data": data
        }
        
        with open(debug_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

**Configuration Validation**:
```python
def validate_configuration():
    """Validate configuration before server startup"""
    config = Config()
    errors = config.validate_for_command("mcp")
    
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)
```

---

## 10. Key Architectural Patterns for Agentic Development

### 1. Protocol-First Design

**Lesson**: Design your agentic tool with the AI integration protocol as a first-class concern.

- **Clean Communication**: Suppress logs and stderr output that interfere with protocol
- **Structured Responses**: Always return structured, parseable data
- **Error Transparency**: Provide helpful error messages with actionable suggestions
- **Tool Abstraction**: Map complex operations to simple, well-documented tool interfaces

### 2. Async-First Architecture

**Lesson**: Use asyncio patterns for responsive tool behavior.

- **Non-Blocking Operations**: Never block user-facing operations with background tasks
- **Priority Queues**: Implement priority-based task coordination
- **Graceful Concurrency**: Use proper locking and coordination primitives
- **Resource Management**: Implement comprehensive cleanup and lifecycle management

### 3. Provider Pattern for Extensibility

**Lesson**: Abstract external dependencies behind clean interfaces.

```python
# Abstract away external services
class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: list[str]) -> EmbeddingResult: ...

class DatabaseProvider(Protocol):
    def search_semantic(self, query_vector: list[float]) -> SearchResult: ...

# Easy to swap providers without changing core logic
providers = {
    "openai": OpenAIEmbeddingProvider(),
    "bge": BGEEmbeddingProvider(),
    "local": LocalEmbeddingProvider()
}
```

### 4. Configuration-Driven Architecture

**Lesson**: Make your tool configurable without code changes.

- **Environment Variables**: Support 12-factor app configuration patterns
- **Configuration Files**: Allow file-based configuration with validation
- **Auto-Detection**: Intelligently detect project structure and context
- **Validation**: Fail fast with clear error messages for configuration issues

### 5. Real-Time Capability with Fallbacks

**Lesson**: Provide real-time features but ensure they degrade gracefully.

- **File Watching**: Monitor changes in real-time when possible
- **Offline Recovery**: Handle situations where real-time monitoring is unavailable
- **Catch-Up Logic**: Process changes that occurred while tool was offline
- **Manual Triggers**: Provide explicit commands to refresh/rebuild when needed

### 6. Production-Ready Error Handling

**Lesson**: Handle errors comprehensively from day one.

- **Structured Errors**: Use consistent error response formats
- **Recovery Mechanisms**: Implement automatic recovery for common failure modes
- **Debug Information**: Provide debugging information without breaking normal operation
- **Graceful Degradation**: Continue operating with reduced functionality when components fail

---

## Summary

ChunkHound demonstrates sophisticated patterns for building **production-ready agentic development tools**:

1. **Protocol Integration**: Clean MCP server implementation with proper JSON-RPC handling
2. **Modular Architecture**: Pluggable providers for databases, embeddings, and parsers
3. **Real-Time Processing**: Sophisticated file watching with priority-based coordination
4. **Robust Error Handling**: Comprehensive error recovery and graceful degradation
5. **Performance Optimization**: Efficient bulk operations and resource management
6. **Configuration Management**: Flexible, validated configuration system

These patterns provide a solid foundation for building sophisticated AI-integrated development tools that are reliable, performant, and maintainable.