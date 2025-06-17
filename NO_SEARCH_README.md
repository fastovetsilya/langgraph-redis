# RedisSaver without RediSearch

This implementation provides a RedisSaver for LangGraph that works with standard Redis installations without requiring the RediSearch module.

## Overview

The original RedisSaver implementation relied on the RediSearch module and `redisvl` (Redis Vector Library) for indexing and querying capabilities. This version replaces that functionality with basic Redis operations that work with any standard Redis installation.

## Changes Made

### 1. Removed RediSearch Dependencies

- Removed imports of `redisvl.index.SearchIndex`
- Removed imports of `redisvl.query.FilterQuery` and related query objects
- Removed `RedisConnectionFactory` from redisvl
- Updated type definitions to not reference SearchIndex types

### 2. Replaced Search Operations with Basic Redis Operations

**Checkpoint Storage:**
- Changed from JSON documents with search indexes to Redis hashes
- Keys follow pattern: `checkpoint:{thread_id}:{checkpoint_ns}:{checkpoint_id}`

**Blob Storage:**
- Changed from indexed JSON documents to Redis hashes
- Keys follow pattern: `checkpoint_blob:{thread_id}:{checkpoint_ns}:{channel}:{version}`

**Write Storage:**
- Changed from indexed JSON documents to Redis hashes  
- Keys follow pattern: `checkpoint_write:{thread_id}:{checkpoint_ns}:{checkpoint_id}:{task_id}:{idx}`

**Querying:**
- Replaced FilterQuery with Redis SCAN operations using key patterns
- Added in-memory filtering for complex queries
- Maintained the same public API for backward compatibility

### 3. Key Implementation Details

**Client Configuration:**
- Simplified Redis client creation without redisvl dependency
- Direct use of `Redis.from_url()` and `RedisCluster.from_url()`
- Maintained cluster mode detection

**Data Serialization:**
- Checkpoint data stored as JSON strings in hash fields
- Blob data stored directly in hash fields
- Metadata stored as JSON strings

**Querying Logic:**
- Used Redis SCAN with pattern matching for key discovery
- Applied filters in-memory after retrieving data
- Maintained sorting and limiting functionality

## API Compatibility

The public API remains unchanged:
- `put()` - Store checkpoints
- `get_tuple()` - Retrieve checkpoints
- `list()` - List checkpoints with optional filtering
- `delete_thread()` - Delete all data for a thread
- `put_writes()` - Store intermediate writes

## Performance Considerations

**Advantages:**
- No dependency on RediSearch module
- Works with any standard Redis installation
- Simpler deployment and maintenance

**Trade-offs:**
- SCAN operations may be slower than indexed searches for large datasets
- In-memory filtering requires transferring more data from Redis
- Less efficient complex queries compared to RediSearch

## Testing

Run the provided test script to verify functionality:

```bash
python test_no_search_redis.py
```

Make sure you have a Redis server running on `localhost:6379` or modify the connection string in the test script.

## Migration from Original Implementation

To migrate from the RediSearch-based implementation:

1. **Data Migration:** The data format has changed, so existing checkpoints will not be compatible
2. **Dependencies:** Remove `redisvl` from your requirements if not used elsewhere
3. **Configuration:** No changes needed to your existing RedisSaver configuration

## Limitations

- No full-text search capabilities
- Complex queries are less efficient than with RediSearch
- SCAN operations may have higher latency on large datasets
- Some advanced filtering capabilities are simplified

## Requirements

- Python >= 3.8
- redis-py >= 4.0.0
- Standard Redis server (no modules required)

## Usage Example

```python
from langgraph.checkpoint.redis import RedisSaver

# Using connection string
with RedisSaver.from_conn_string("redis://localhost:6379") as saver:
    saver.setup()
    # Use saver for checkpoint operations

# Using existing Redis client
import redis
client = redis.Redis(host="localhost", port=6379)
with RedisSaver.from_conn_string(redis_client=client) as saver:
    saver.setup()
    # Use saver for checkpoint operations
```