#!/usr/bin/env python3
"""
Debug script to test Redis cluster connectivity and identify issues.
"""

import logging
from typing import Any, Dict
from langgraph.checkpoint.redis import RedisSaver

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_cluster_connection():
    """Test various cluster connection methods."""
    
    print("🔍 Testing Redis cluster connectivity...")
    
    # Test configurations - adjust these to match your cluster
    test_configs = [
        {
            "name": "Single URL with cluster_mode",
            "redis_url": "redis://10.103.33.103:6381",  # Replace with your cluster IP
            "connection_args": {"cluster_mode": True}
        },
        {
            "name": "Multiple startup nodes",
            "redis_url": None,
            "connection_args": {
                "cluster_mode": True,
                "startup_nodes": [
                    {"host": "10.103.33.101", "port": 6381},
                    {"host": "10.103.33.102", "port": 6381},
                    {"host": "10.103.33.103", "port": 6381}
                ]
            }
        },
        {
            "name": "Single node only",
            "redis_url": None,
            "connection_args": {
                "cluster_mode": True,
                "startup_nodes": [
                    {"host": "10.103.33.103", "port": 6381}
                ]
            }
        }
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {config['name']}")
        print(f"URL: {config['redis_url']}")
        print(f"Args: {config['connection_args']}")
        print(f"{'='*60}")
        
        try:
            with RedisSaver.from_conn_string(
                redis_url=config["redis_url"],
                connection_args=config["connection_args"]
            ) as checkpointer:
                print("✅ RedisSaver created successfully")
                
                # Test setup
                checkpointer.setup()
                print(f"✅ Setup completed - cluster_mode: {checkpointer.cluster_mode}")
                
                # Test basic operation
                thread_id = f"debug-test-{i}"
                test_config = {"configurable": {"thread_id": thread_id}}
                
                from langgraph.checkpoint.base import empty_checkpoint
                test_checkpoint = empty_checkpoint()
                test_metadata = {"source": "debug", "step": 1}
                
                # Test put
                checkpointer.put(test_config, test_checkpoint, test_metadata, {})
                print("✅ Put operation successful")
                
                # Test get
                retrieved = checkpointer.get_tuple(test_config)
                if retrieved:
                    print("✅ Get operation successful")
                else:
                    print("⚠️ Get operation returned None")
                
                # Test list
                checkpoints = list(checkpointer.list(test_config))
                print(f"✅ List operation successful - found {len(checkpoints)} checkpoints")
                
                # Cleanup
                checkpointer.delete_thread(thread_id)
                print("✅ Cleanup successful")
                
                print(f"🎉 Test {i+1} PASSED!")
                
        except Exception as e:
            print(f"❌ Test {i+1} FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()

def test_direct_redis_cluster():
    """Test direct redis-py cluster connection."""
    print("\n🔍 Testing direct redis-py cluster connection...")
    
    try:
        from redis.cluster import RedisCluster
        
        # Test direct cluster connection
        startup_nodes = [
            {"host": "10.103.33.101", "port": 6381},
            {"host": "10.103.33.102", "port": 6381},
            {"host": "10.103.33.103", "port": 6381}
        ]
        
        print(f"Connecting to cluster with nodes: {startup_nodes}")
        
        cluster = RedisCluster(startup_nodes=startup_nodes, decode_responses=True)
        
        # Test ping
        result = cluster.ping()
        print(f"✅ Cluster ping successful: {result}")
        
        # Test cluster info
        try:
            cluster_info = cluster.cluster("info")
            print(f"✅ Cluster info: {cluster_info}")
        except Exception as e:
            print(f"⚠️ Cluster info failed: {e}")
        
        # Test basic operations
        cluster.set("debug:test", "hello")
        value = cluster.get("debug:test")
        print(f"✅ Basic set/get successful: {value}")
        
        cluster.delete("debug:test")
        print("✅ Direct cluster test PASSED!")
        
        cluster.close()
        
    except Exception as e:
        print(f"❌ Direct cluster test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Redis Cluster Debug Tool")
    print("📝 Make sure to update the IP addresses in this script to match your cluster!")
    print()
    
    # Test direct cluster connection first
    test_direct_redis_cluster()
    
    # Test RedisSaver with cluster
    test_cluster_connection()
    
    print("\n" + "="*60)
    print("🏁 Debug tests completed!")
    print("\n💡 Tips:")
    print("  - If direct cluster test fails, check your cluster configuration")
    print("  - If RedisSaver tests fail but direct cluster works, there may be an initialization issue")
    print("  - Check that all cluster nodes are accessible from your client")
    print("  - Verify cluster is properly configured with 'cluster-enabled yes'") 