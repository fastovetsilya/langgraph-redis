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
    
    # IMPORTANT: Update these credentials with your actual values
    REDIS_USERNAME = "your_username"  # Replace with your Redis username
    REDIS_PASSWORD = "your_password"  # Replace with your Redis password
    REDIS_HOSTS = [
        "redis-1.example.net",  # Replace with your actual FQDNs
        "redis-2.example.net", 
        "redis-3.example.net"
    ]
    REDIS_PORT = 6379
    
    # Test configurations - UPDATED for your FQDN setup with auth
    test_configs = [
        {
            "name": "Single FQDN with auth in URL",
            "redis_url": f"redis://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOSTS[0]}:{REDIS_PORT}",
            "connection_args": {"cluster_mode": True}
        },
        {
            "name": "Multiple startup nodes with auth in connection_args",
            "redis_url": None,
            "connection_args": {
                "cluster_mode": True,
                "username": REDIS_USERNAME,
                "password": REDIS_PASSWORD,
                "startup_nodes": [
                    {"host": host, "port": REDIS_PORT} for host in REDIS_HOSTS
                ]
            }
        },
        {
            "name": "Single FQDN with auth in connection_args", 
            "redis_url": f"redis://{REDIS_HOSTS[0]}:{REDIS_PORT}",
            "connection_args": {
                "cluster_mode": True,
                "username": REDIS_USERNAME,
                "password": REDIS_PASSWORD
            }
        },
        {
            "name": "Single node only with auth",
            "redis_url": None,
            "connection_args": {
                "cluster_mode": True,
                "username": REDIS_USERNAME,
                "password": REDIS_PASSWORD,
                "startup_nodes": [
                    {"host": REDIS_HOSTS[0], "port": REDIS_PORT}
                ]
            }
        },
        {
            "name": "Standalone mode test with auth (fallback)",
            "redis_url": f"redis://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOSTS[0]}:{REDIS_PORT}",
            "connection_args": {}  # No cluster_mode
        }
    ]
    
    print(f"🔐 Testing with username: {REDIS_USERNAME}")
    print(f"🏠 Testing with hosts: {REDIS_HOSTS}")
    print("⚠️  Make sure to update REDIS_USERNAME, REDIS_PASSWORD, and REDIS_HOSTS above!")
    print()
    
    for i, config in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {config['name']}")
        # Don't print passwords in logs
        safe_url = config['redis_url'].replace(REDIS_PASSWORD, '***') if config['redis_url'] and REDIS_PASSWORD in config['redis_url'] else config['redis_url']
        print(f"URL: {safe_url}")
        safe_args = config['connection_args'].copy()
        if 'password' in safe_args:
            safe_args['password'] = '***'
        print(f"Args: {safe_args}")
        print(f"{'='*60}")
        
        saver = None
        try:
            # Step 1: Create RedisSaver
            print("Step 1: Creating RedisSaver...")
            saver = RedisSaver(
                redis_url=config["redis_url"],
                connection_args=config["connection_args"]
            )
            print("✅ RedisSaver created successfully")
            
            # Step 2: Check if _redis is properly initialized
            print("Step 2: Checking Redis client initialization...")
            if hasattr(saver, '_redis') and saver._redis is not None:
                print(f"✅ Redis client initialized: {type(saver._redis)}")
            else:
                print("❌ Redis client is None - this is likely causing your error!")
                print("  - Check if configure_client() was called")
                print("  - Check connection parameters")
                continue
            
            # Step 3: Test setup
            print("Step 3: Running setup...")
            saver.setup()
            print(f"✅ Setup completed - cluster_mode: {getattr(saver, 'cluster_mode', 'Unknown')}")
            
            # Step 4: Test ping
            print("Step 4: Testing ping...")
            ping_result = saver._redis.ping()
            print(f"✅ Ping successful: {ping_result}")
            
            # Step 5: Test basic operation
            print("Step 5: Testing basic operations...")
            thread_id = f"debug-test-{i}"
            test_config = {"configurable": {"thread_id": thread_id}}
            
            from langgraph.checkpoint.base import empty_checkpoint
            test_checkpoint = empty_checkpoint()
            test_metadata = {"source": "debug", "step": 1}
            
            # Test put
            saver.put(test_config, test_checkpoint, test_metadata, {})
            print("✅ Put operation successful")
            
            # Test get
            retrieved = saver.get_tuple(test_config)
            if retrieved:
                print("✅ Get operation successful")
            else:
                print("⚠️ Get operation returned None")
            
            # Test list
            checkpoints = list(saver.list(test_config))
            print(f"✅ List operation successful - found {len(checkpoints)} checkpoints")
            
            # Cleanup
            saver.delete_thread(thread_id)
            print("✅ Cleanup successful")
            
            print(f"🎉 Test {i+1} PASSED!")
            break  # Stop on first successful test
            
        except AttributeError as e:
            if "redis_connection" in str(e):
                print(f"❌ Test {i+1} FAILED with redis_connection error: {e}")
                print("🔍 Debugging redis_connection issue:")
                if saver:
                    print(f"  - saver._redis exists: {hasattr(saver, '_redis')}")
                    print(f"  - saver._redis value: {getattr(saver, '_redis', 'NOT_SET')}")
                    print(f"  - saver._owns_its_client: {getattr(saver, '_owns_its_client', 'NOT_SET')}")
            else:
                print(f"❌ Test {i+1} FAILED with AttributeError: {e}")
            import traceback
            traceback.print_exc()
            print()
        except Exception as e:
            print(f"❌ Test {i+1} FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()
        finally:
            # Clean up
            if saver and hasattr(saver, '_redis') and saver._redis and getattr(saver, '_owns_its_client', False):
                try:
                    saver._redis.close()
                except:
                    pass

def test_direct_redis_cluster():
    """Test direct redis-py cluster connection."""
    print("\n🔍 Testing direct redis-py cluster connection...")
    
    # IMPORTANT: Update these credentials with your actual values
    REDIS_USERNAME = "your_username"  # Replace with your Redis username  
    REDIS_PASSWORD = "your_password"  # Replace with your Redis password
    REDIS_HOSTS = [
        "redis-1.example.net",  # Replace with your actual FQDNs
        "redis-2.example.net",
        "redis-3.example.net"
    ]
    REDIS_PORT = 6379
    
    try:
        from redis.cluster import RedisCluster
        
        # Test direct cluster connection with your FQDNs and auth
        startup_nodes = [
            {"host": host, "port": REDIS_PORT} for host in REDIS_HOSTS
        ]
        
        print(f"Connecting to cluster with nodes: {[f'{host}:{REDIS_PORT}' for host in REDIS_HOSTS]}")
        print("NOTE: Replace the example.net hostnames and credentials with your actual values!")
        
        cluster = RedisCluster(
            startup_nodes=startup_nodes, 
            decode_responses=True,
            username=REDIS_USERNAME,
            password=REDIS_PASSWORD
        )
        
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
        print("💡 Common issues:")
        print("  - Check that the hostnames resolve correctly")
        print("  - Verify the port (6379) is correct")
        print("  - Check username and password are correct")
        print("  - Ensure Redis cluster is running and accessible")
        print("  - Check firewall/network connectivity")
        print("  - Verify Redis AUTH is configured properly")
        import traceback
        traceback.print_exc()

def test_context_manager():
    """Test using the context manager approach."""
    print("\n🔍 Testing context manager approach...")
    
    # IMPORTANT: Update these credentials
    REDIS_USERNAME = "your_username"
    REDIS_PASSWORD = "your_password"  
    REDIS_HOST = "redis-1.example.net"
    REDIS_PORT = 6379
    
    test_configs = [
        {
            "name": "Context manager with auth in URL",
            "redis_url": f"redis://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}",
            "connection_args": {"cluster_mode": True}
        },
        {
            "name": "Context manager with auth in connection_args",
            "redis_url": f"redis://{REDIS_HOST}:{REDIS_PORT}",
            "connection_args": {
                "cluster_mode": True,
                "username": REDIS_USERNAME,
                "password": REDIS_PASSWORD
            }
        }
    ]
    
    for config in test_configs:
        print(f"Testing: {config['name']}")
        try:
            with RedisSaver.from_conn_string(
                redis_url=config["redis_url"],
                connection_args=config["connection_args"]
            ) as checkpointer:
                print("✅ Context manager created successfully")
                
                # Test setup
                checkpointer.setup()
                print(f"✅ Setup completed - cluster_mode: {getattr(checkpointer, 'cluster_mode', 'Unknown')}")
                
                # Quick test
                thread_id = "ctx-test"
                test_config = {"configurable": {"thread_id": thread_id}}
                
                from langgraph.checkpoint.base import empty_checkpoint
                checkpointer.put(test_config, empty_checkpoint(), {}, {})
                print("✅ Context manager test PASSED!")
                
                checkpointer.delete_thread(thread_id)
                break  # Stop on first success
                
        except Exception as e:
            print(f"❌ Context manager test FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Redis Cluster Debug Tool with Authentication")
    print("📝 IMPORTANT: Update the credentials and hostnames in this script!")
    print("   1. Replace 'your_username' and 'your_password' with your Redis credentials")
    print("   2. Replace 'redis-1.example.net', etc. with your actual Redis FQDNs")
    print("   3. Your setup uses port 6379 (already configured)")
    print()
    print("🔐 Authentication Note:")
    print("   - You can use multiple nodes for initial connection discovery")
    print("   - The same username/password will be used for all cluster nodes")
    print("   - Redis will automatically discover all cluster nodes after connecting to any one")
    print()
    
    # Test direct cluster connection first
    test_direct_redis_cluster()
    
    # Test RedisSaver with cluster
    test_cluster_connection()
    
    # Test context manager approach
    test_context_manager()
    
    print("\n" + "="*60)
    print("🏁 Debug tests completed!")
    print("\n💡 Tips for your authenticated setup:")
    print("  - Update the script with your actual Redis credentials and hostnames")
    print("  - You can use multiple startup nodes - they all use the same auth")
    print("  - If direct cluster test fails, check credentials and network connectivity")
    print("  - If RedisSaver tests fail but direct cluster works, there may be an initialization issue")
    print("  - The 'redis_connection' error suggests the Redis client is None during initialization")
    print("  - Try the context manager approach as it handles cleanup automatically")
    print("  - Verify cluster is properly configured with 'cluster-enabled yes'")
    print("  - Make sure AUTH is configured on all cluster nodes") 