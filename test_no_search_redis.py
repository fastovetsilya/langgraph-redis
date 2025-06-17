#!/usr/bin/env python3
"""
Simple test script to verify RedisSaver works without RediSearch.
Run this script to test basic checkpoint functionality.
"""

import json
from typing import Any, Dict
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

# Import the new RedisSaver
from langgraph.checkpoint.redis import RedisSaver


def test_basic_functionality():
    """Test basic RedisSaver functionality without RediSearch."""
    
    # Use default Redis connection (localhost:6379)
    redis_url = "redis://localhost:6379"
    
    print("🧪 Testing RedisSaver without RediSearch...")
    
    try:
        with RedisSaver.from_conn_string(redis_url) as saver:
            saver.setup()
            print("✅ Successfully created RedisSaver")
            
            # Test 1: Basic checkpoint storage and retrieval
            print("\n📝 Test 1: Basic checkpoint operations")
            
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": f"test-thread-{uuid4()}",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1",
                }
            }
            
            checkpoint: Checkpoint = empty_checkpoint()
            metadata: CheckpointMetadata = {
                "source": "test",
                "step": 1,
                "writes": {},
            }
            
            # Store checkpoint
            stored_config = saver.put(config, checkpoint, metadata, {})
            print(f"✅ Stored checkpoint: {stored_config['configurable']['checkpoint_id']}")
            
            # Retrieve checkpoint
            retrieved = saver.get_tuple(stored_config)
            assert retrieved is not None, "Failed to retrieve checkpoint"
            assert retrieved.metadata["source"] == "test"
            print("✅ Retrieved checkpoint successfully")
            
            # Test 2: List checkpoints
            print("\n📝 Test 2: List checkpoints")
            
            # Store a second checkpoint
            config2: RunnableConfig = {
                "configurable": {
                    "thread_id": config["configurable"]["thread_id"],
                    "checkpoint_ns": "",
                    "checkpoint_id": "2",
                }
            }
            
            checkpoint2 = create_checkpoint(checkpoint, {}, 2)
            metadata2: CheckpointMetadata = {
                "source": "loop",
                "step": 2,
                "writes": {},
            }
            
            saver.put(config2, checkpoint2, metadata2, {})
            print("✅ Stored second checkpoint")
            
            # List all checkpoints for this thread
            checkpoints = list(saver.list(config))
            assert len(checkpoints) >= 2, f"Expected at least 2 checkpoints, got {len(checkpoints)}"
            print(f"✅ Listed {len(checkpoints)} checkpoints")
            
            # Test 3: Filter checkpoints
            print("\n📝 Test 3: Filter checkpoints")
            
            filtered = list(saver.list(None, filter={"source": "test"}))
            assert len(filtered) >= 1, "Filter by source failed"
            print(f"✅ Filtered checkpoints by source: found {len(filtered)}")
            
            filtered_step = list(saver.list(None, filter={"step": 1}))
            assert len(filtered_step) >= 1, "Filter by step failed"
            print(f"✅ Filtered checkpoints by step: found {len(filtered_step)}")
            
            # Test 4: Channel values
            print("\n📝 Test 4: Channel values")
            
            # Create checkpoint with channel values
            config3: RunnableConfig = {
                "configurable": {
                    "thread_id": f"test-thread-{uuid4()}",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1",
                }
            }
            
            channel_values = {"messages": ["Hello", "World"], "count": 42}
            checkpoint3 = empty_checkpoint()
            checkpoint3["channel_values"] = channel_values
            
            new_versions = {"messages": "1.0", "count": "1.0"}
            stored_config3 = saver.put(config3, checkpoint3, {}, new_versions)
            print("✅ Stored checkpoint with channel values")
            
            # Retrieve and check channel values
            retrieved3 = saver.get_tuple(stored_config3)
            assert retrieved3 is not None, "Failed to retrieve checkpoint with channel values"
            print("✅ Retrieved checkpoint with channel values")
            
            # Test 5: Delete thread
            print("\n📝 Test 5: Delete thread")
            
            thread_id = config["configurable"]["thread_id"]
            saver.delete_thread(thread_id)
            print(f"✅ Deleted thread: {thread_id}")
            
            # Verify deletion
            remaining = list(saver.list(config))
            assert len(remaining) == 0, f"Thread deletion failed, {len(remaining)} checkpoints remain"
            print("✅ Verified thread deletion")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All tests passed! RedisSaver works without RediSearch.")
    return True


if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)