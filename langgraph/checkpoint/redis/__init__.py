from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.constants import TASKS
from redis import Redis
from redis.cluster import RedisCluster

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.base import BaseRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver
from langgraph.checkpoint.redis.util import (
    EMPTY_ID_SENTINEL,
    from_storage_safe_id,
    from_storage_safe_str,
    to_storage_safe_id,
    to_storage_safe_str,
)
from langgraph.checkpoint.redis.version import __lib_name__, __version__

logger = logging.getLogger(__name__)


class RedisSaver(BaseRedisSaver[Union[Redis, RedisCluster], None]):
    """Redis implementation for checkpoint saving without RediSearch."""

    _redis: Union[Redis, RedisCluster]  # Support both standalone and cluster clients
    # Whether to assume the Redis server is a cluster; None triggers auto-detection
    cluster_mode: Optional[bool] = None
    # Store the original startup_nodes for cluster mode scanning fallback
    _startup_nodes: List[dict[str, Any]] = []

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Union[Redis, RedisCluster]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        ttl: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            redis_url=redis_url,
            redis_client=redis_client,
            connection_args=connection_args,
            ttl=ttl,
        )

    def configure_client(
        self,
        redis_url: Optional[str] = None,
        redis_client: Optional[Union[Redis, RedisCluster]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Configure the Redis client."""
        self._owns_its_client = redis_client is None
        # Reset startup nodes list on each configuration
        self._startup_nodes = []
        
        if redis_client:
            self._redis = redis_client
        else:
            # Create Redis client without redisvl dependency
            connection_args = connection_args or {}
            
            # Remove cluster_mode from connection_args as it's not a Redis connection parameter
            cluster_mode_hint = connection_args.pop('cluster_mode', None)
            
            # Extract password and SSL settings for later standalone node scans (if provided)
            standalone_password = connection_args.get('password')
            # Extract SSL configuration for cluster fallback connections
            ssl_config = {}
            if 'ssl' in connection_args:
                ssl_config['ssl'] = connection_args['ssl']
            if 'ssl_cert_reqs' in connection_args:
                ssl_config['ssl_cert_reqs'] = connection_args['ssl_cert_reqs']
            if 'ssl_ca_certs' in connection_args:
                ssl_config['ssl_ca_certs'] = connection_args['ssl_ca_certs']
            if 'ssl_certfile' in connection_args:
                ssl_config['ssl_certfile'] = connection_args['ssl_certfile']
            if 'ssl_keyfile' in connection_args:
                ssl_config['ssl_keyfile'] = connection_args['ssl_keyfile']
            
            # Check for SSL URL (rediss://) or cluster mode
            is_ssl_url = redis_url and redis_url.startswith('rediss://')
            is_cluster_mode = 'cluster' in (redis_url or '').lower() or cluster_mode_hint
            
            if redis_url:
                # Parse URL and create appropriate client
                if is_cluster_mode or is_ssl_url:
                    # Ensure startup_nodes are in the correct format if provided
                    startup_nodes = connection_args.get('startup_nodes', [])
                    if startup_nodes and isinstance(startup_nodes[0], dict):
                        # Convert dict format to ClusterNode format
                        from redis.cluster import ClusterNode
                        formatted_nodes = []
                        for node in startup_nodes:
                            if isinstance(node, dict) and 'host' in node and 'port' in node:
                                # Keep dict version for fallback scans including SSL config
                                node_config = {
                                    'host': node['host'], 
                                    'port': node['port'], 
                                    'password': standalone_password
                                }
                                # Add SSL configuration to fallback node config
                                node_config.update(ssl_config)
                                self._startup_nodes.append(node_config)
                                formatted_nodes.append(ClusterNode(node['host'], node['port']))
                            else:
                                formatted_nodes.append(node)
                        connection_args['startup_nodes'] = formatted_nodes
                    elif startup_nodes:
                        # startup_nodes already ClusterNode instances – preserve for fallback
                        from redis.cluster import ClusterNode
                        for sn in startup_nodes:
                            if isinstance(sn, ClusterNode):
                                node_config = {
                                    'host': sn.host, 
                                    'port': sn.port, 
                                    'password': standalone_password
                                }
                                # Add SSL configuration to fallback node config
                                node_config.update(ssl_config)
                                self._startup_nodes.append(node_config)
                    elif is_ssl_url:
                        # For SSL URLs without explicit startup_nodes, try to parse URL for host/port
                        import urllib.parse
                        try:
                            parsed = urllib.parse.urlparse(redis_url)
                            if parsed.hostname and parsed.port:
                                node_config = {
                                    'host': parsed.hostname,
                                    'port': parsed.port,
                                    'password': parsed.password or standalone_password
                                }
                                # For SSL URLs, ensure SSL is enabled in fallback config
                                if is_ssl_url:
                                    node_config['ssl'] = True
                                    # Inherit SSL settings from connection_args
                                    node_config.update(ssl_config)
                                self._startup_nodes.append(node_config)
                        except Exception as e:
                            logger.warning(f"Failed to parse Redis URL for fallback nodes: {e}")
                    
                    try:
                        if is_cluster_mode:
                            self._redis = RedisCluster.from_url(redis_url, **connection_args)
                        else:
                            # For SSL URLs that aren't explicitly cluster mode, try cluster first
                            # then fallback to standalone
                            try:
                                # Try as cluster first for SSL URLs
                                self._redis = RedisCluster.from_url(redis_url, **connection_args)
                            except Exception:
                                # Fallback to standalone SSL connection
                                self._redis = Redis.from_url(redis_url, **connection_args)
                        # Test the connection
                        self._redis.ping()
                    except Exception as e:
                        logger.warning(f"Failed to create cluster client from URL: {e}")
                        # Fallback to standalone client if cluster fails
                        logger.info("Falling back to standalone Redis client")
                        self._redis = Redis.from_url(redis_url, **{k: v for k, v in connection_args.items() if k != 'startup_nodes'})
                        # Override cluster mode since we're using standalone
                        self.cluster_mode = False
                else:
                    self._redis = Redis.from_url(redis_url, **connection_args)
            else:
                # Default connection
                if cluster_mode_hint:
                    # For cluster mode, we need startup_nodes
                    if 'startup_nodes' not in connection_args:
                        connection_args['startup_nodes'] = [{'host': 'localhost', 'port': 6379}]
                    
                    # Preserve for fallback scans
                    for node in connection_args['startup_nodes']:
                        if isinstance(node, dict):
                            node_config = {
                                'host': node['host'], 
                                'port': node['port'], 
                                'password': standalone_password
                            }
                            # Add SSL configuration to fallback node config
                            node_config.update(ssl_config)
                            self._startup_nodes.append(node_config)
                    
                    try:
                        self._redis = RedisCluster(**connection_args)
                        # Test the connection
                        self._redis.ping()
                    except Exception as e:
                        logger.warning(f"Failed to create cluster client: {e}")
                        # Fallback to standalone client using first startup node
                        if self._startup_nodes:
                            first_node = self._startup_nodes[0]
                            logger.info(f"Falling back to standalone Redis client at {first_node['host']}:{first_node['port']}")
                            fallback_args = {
                                'host': first_node['host'],
                                'port': first_node['port'],
                            }
                            if first_node.get('password'):
                                fallback_args['password'] = first_node['password']
                            # Add SSL configuration to fallback args
                            for ssl_key, ssl_value in ssl_config.items():
                                if ssl_key in first_node:
                                    fallback_args[ssl_key] = first_node[ssl_key]
                            self._redis = Redis(**fallback_args)
                            # Override cluster mode since we're using standalone
                            self.cluster_mode = False
                        else:
                            logger.info("Falling back to default standalone Redis client")
                            self._redis = Redis(**{k: v for k, v in connection_args.items() if k != 'startup_nodes'})
                            # Override cluster mode since we're using standalone
                            self.cluster_mode = False
                else:
                    self._redis = Redis(**connection_args)

    def create_indexes(self) -> None:
        """No-op since we don't use search indexes."""
        pass

    def setup(self) -> None:
        """Initialize the Redis connection and detect cluster mode."""
        self._detect_cluster_mode()
        # No need to create search indexes

    def _detect_cluster_mode(self) -> None:
        """Detect if the Redis client is a cluster client by inspecting its class."""
        if self.cluster_mode is not None:
            logger.info(
                f"Redis cluster_mode explicitly set to {self.cluster_mode}, skipping detection."
            )
            return

        # Determine cluster mode based on client class
        if isinstance(self._redis, RedisCluster):
            logger.info("Redis client is a cluster client candidate, validating...")
            if self._is_cluster_healthy():
                logger.info("Cluster client is healthy, enabling cluster_mode")
                self.cluster_mode = True
                return
            else:
                logger.warning("Cluster client validation failed, attempting recovery with standalone client")
                self._attempt_cluster_recovery()
                return
        else:
            logger.info("Redis client is a standalone client")
            self.cluster_mode = False

    def _attempt_cluster_recovery(self) -> None:
        """Attempt to recover from a broken cluster client by creating a standalone client."""
        logger.info("Attempting to recover from broken cluster client...")
        
        # STEP 1 ──────────────────────────────────────────────────────────
        # Try to rebuild a brand-new RedisCluster client first.  A MOVED/ASK
        # error often means we accidentally downgraded to a standalone client
        # that is now talking to a cluster node.  If we can successfully
        # recreate a healthy cluster client, we prefer that over falling back
        # to a standalone connection.

        if hasattr(self, "_startup_nodes") and self._startup_nodes:
            for node in self._startup_nodes:
                try:
                    host = node.get("host")
                    port = node.get("port")
                    if not host or not port:
                        continue

                    logger.info(
                        f"Attempting to recreate RedisCluster client via {host}:{port}"
                    )

                    # Build minimal args for RedisCluster
                    cluster_args: dict[str, Any] = {
                        "startup_nodes": [{"host": host, "port": port}],
                        "socket_timeout": 10,
                        "socket_connect_timeout": 10,
                    }

                    # Common optional parameters
                    if node.get("password"):
                        cluster_args["password"] = node["password"]

                    # Propagate SSL related keys if present
                    ssl_keys = [
                        "ssl",
                        "ssl_cert_reqs",
                        "ssl_ca_certs",
                        "ssl_certfile",
                        "ssl_keyfile",
                    ]
                    for ssl_key in ssl_keys:
                        if ssl_key in node:
                            cluster_args[ssl_key] = node[ssl_key]

                    from redis.cluster import RedisCluster

                    new_cluster_client = RedisCluster(**cluster_args)
                    # Quick health-check
                    new_cluster_client.ping()

                    logger.info(
                        f"Successfully recreated RedisCluster client via {host}:{port}"
                    )

                    # Close previous client if we created it internally
                    if self._owns_its_client and hasattr(self._redis, "close"):
                        try:
                            self._redis.close()
                        except Exception:
                            pass

                    self._redis = new_cluster_client
                    self.cluster_mode = True
                    logger.info("Recovered with refreshed RedisCluster client")
                    return  # SUCCESS – no need to try standalone

                except Exception as recreate_exc:
                    logger.warning(
                        f"Failed to recreate RedisCluster client for {node}: {recreate_exc}"
                    )
                    continue

        # STEP 2 ──────────────────────────────────────────────────────────
        # If we reach here, recreating a cluster client failed.  Fall back to
        # the original behaviour of trying a direct standalone connection.
        logger.info("Attempting to recover from broken cluster client...")
        
        # Try to create a standalone client using the first startup node
        if hasattr(self, '_startup_nodes') and self._startup_nodes:
            for node in self._startup_nodes:
                try:
                    host = node.get('host')
                    port = node.get('port')
                    if not host or not port:
                        continue
                        
                    logger.info(f"Attempting to create standalone client for {host}:{port}")
                    
                    # Build connection args for standalone client
                    standalone_args = {
                        'host': host,
                        'port': port,
                        'socket_timeout': 10,
                        'socket_connect_timeout': 10,
                    }
                    
                    # Add credentials if available
                    if node.get('password'):
                        standalone_args['password'] = node['password']
                    
                    # Add SSL configuration if present
                    ssl_keys = ['ssl', 'ssl_cert_reqs', 'ssl_ca_certs', 'ssl_certfile', 'ssl_keyfile']
                    for ssl_key in ssl_keys:
                        if ssl_key in node:
                            standalone_args[ssl_key] = node[ssl_key]
                    
                    # Create standalone client
                    standalone_client = Redis(**standalone_args)
                    
                    # Test the connection
                    standalone_client.ping()
                    
                    # If we get here, the standalone client works
                    logger.info(f"Successfully created standalone client for {host}:{port}")
                    
                    # Replace the broken cluster client
                    if self._owns_its_client and hasattr(self._redis, 'close'):
                        try:
                            self._redis.close()
                        except:
                            pass
                    
                    self._redis = standalone_client
                    self.cluster_mode = False
                    logger.info("Successfully recovered with standalone Redis client")
                    return
                    
                except Exception as e:
                    logger.warning(f"Failed to create standalone client for {node}: {e}")
                    continue
        
        # If we can't recover, at least set cluster_mode to False
        logger.warning("Could not recover with standalone client. Treating as broken cluster client.")
        self.cluster_mode = False

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from Redis using basic key operations."""
        # Build search pattern
        thread_id = None
        checkpoint_ns = ""
        
        if config:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            # Note: For list operations, we typically want all checkpoints for a thread,
            # regardless of any checkpoint_id in the config

        # Create key pattern
        if thread_id:
            pattern_parts = [
                "checkpoint",
                to_storage_safe_id(thread_id),
                to_storage_safe_str(checkpoint_ns),
                "*"  # Always use wildcard for listing all checkpoints in thread
            ]
            pattern = ":".join(pattern_parts)
        else:
            pattern = "checkpoint:*"

        # Use cluster-safe key scanning
        keys = self._scan_keys_cluster_safe(pattern, count=1000)
        # Convert to bytes for consistency with original code
        keys = [k.encode() if isinstance(k, str) else k for k in keys]
        
        # Limit keys if specified
        if limit:
            keys = keys[:limit]

        # Process each key
        for key_bytes in keys:
            key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
            
            try:
                # Parse key to get thread_id, checkpoint_ns, checkpoint_id
                parts = key.split(":")
                if len(parts) < 4 or parts[0] != "checkpoint":
                    continue
                    
                key_thread_id = from_storage_safe_id(parts[1])
                key_checkpoint_ns = from_storage_safe_str(parts[2])
                key_checkpoint_id = from_storage_safe_id(parts[3])
                
                # Apply additional filters
                if filter:
                    checkpoint_data = self._redis.hgetall(key)
                    if not checkpoint_data:
                        continue
                        
                    # Check filter conditions
                    if "source" in filter:
                        if checkpoint_data.get(b"source", b"").decode() != filter["source"]:
                            continue
                    if "step" in filter:
                        try:
                            step_value = int(checkpoint_data.get(b"step", b"0"))
                            if step_value != filter["step"]:
                                continue
                        except (ValueError, TypeError):
                            continue

                # Get checkpoint data
                checkpoint_data = self._redis.hgetall(key)
                if not checkpoint_data:
                    continue

                # Decode bytes keys/values
                decoded_data = {}
                for k, v in checkpoint_data.items():
                    key_str = k.decode() if isinstance(k, bytes) else k
                    val_str = v.decode() if isinstance(v, bytes) else v
                    decoded_data[key_str] = val_str

                # Get channel values
                channel_values = self.get_channel_values(
                    thread_id=key_thread_id,
                    checkpoint_ns=key_checkpoint_ns,
                    checkpoint_id=key_checkpoint_id,
                )

                # Get pending sends from parent checkpoint
                pending_sends = []
                parent_checkpoint_id = decoded_data.get("parent_checkpoint_id")
                if parent_checkpoint_id and parent_checkpoint_id != EMPTY_ID_SENTINEL:
                    pending_sends = self._load_pending_sends(
                        thread_id=key_thread_id,
                        checkpoint_ns=key_checkpoint_ns,
                        parent_checkpoint_id=from_storage_safe_id(parent_checkpoint_id),
                    )

                # Parse metadata
                metadata_str = decoded_data.get("metadata", "{}")
                try:
                    metadata_dict = json.loads(metadata_str)
                except (json.JSONDecodeError, TypeError):
                    metadata_dict = {}

                # Sanitize metadata
                sanitized_metadata = {
                    k.replace("\u0000", ""): (
                        v.replace("\u0000", "") if isinstance(v, str) else v
                    )
                    for k, v in metadata_dict.items()
                }
                metadata = cast(CheckpointMetadata, sanitized_metadata)

                config_param: RunnableConfig = {
                    "configurable": {
                        "thread_id": key_thread_id,
                        "checkpoint_ns": key_checkpoint_ns,
                        "checkpoint_id": key_checkpoint_id,
                    }
                }

                checkpoint_param = self._load_checkpoint(
                    decoded_data.get("checkpoint", "{}"),
                    channel_values,
                    pending_sends,
                )

                pending_writes = self._load_pending_writes(
                    key_thread_id, key_checkpoint_ns, key_checkpoint_id
                )

                yield CheckpointTuple(
                    config=config_param,
                    checkpoint=checkpoint_param,
                    metadata=metadata,
                    parent_config=None,
                    pending_writes=pending_writes,
                )
                
            except Exception as e:
                logger.warning(f"Error processing checkpoint key {key}: {e}")
                continue

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint to Redis using hash operations."""
        configurable = config["configurable"].copy()

        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns", "")
        thread_ts = configurable.pop("thread_ts", "")
        checkpoint_id = (
            configurable.pop("checkpoint_id", configurable.pop("thread_ts", ""))
            or thread_ts
        )

        # For values we store in Redis, we need to convert empty strings to the
        # sentinel value.
        storage_safe_thread_id = to_storage_safe_id(thread_id)
        storage_safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        storage_safe_checkpoint_id = to_storage_safe_id(checkpoint_id)

        copy = checkpoint.copy()
        # When we return the config, we need to preserve empty strings that
        # were passed in, instead of the sentinel value.
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

        # Create the checkpoint key
        checkpoint_key = BaseRedisSaver._make_redis_checkpoint_key(
            storage_safe_thread_id,
            storage_safe_checkpoint_ns,
            storage_safe_checkpoint_id,
        )

        # Store checkpoint data as hash
        checkpoint_data = {
            "thread_id": storage_safe_thread_id,
            "checkpoint_ns": storage_safe_checkpoint_ns,
            "checkpoint_id": storage_safe_checkpoint_id,
            "parent_checkpoint_id": storage_safe_checkpoint_id,
            "checkpoint": json.dumps(self._dump_checkpoint(copy)),
            "metadata": self._dump_metadata(metadata),
        }

        # Add filter fields if they exist in metadata
        if all(key in metadata for key in ["source", "step"]):
            checkpoint_data["source"] = str(metadata["source"]) if metadata["source"] is not None else ""
            checkpoint_data["step"] = str(metadata["step"]) if metadata["step"] is not None else "0"

        # Clean checkpoint_data to ensure no None values
        cleaned_checkpoint_data = {}
        for k, v in checkpoint_data.items():
            if v is not None:
                cleaned_checkpoint_data[k] = v
            else:
                cleaned_checkpoint_data[k] = ""

        # Health-check: if we're still holding a RedisCluster client but it has become
        # unusable (e.g. get_default_node returns None), attempt to recover by
        # falling back to a standalone connection. This prevents errors such as
        # "NoneType object has no attribute 'redis_connection'" that originate
        # from redis-py internals when the cluster metadata is missing.
        if isinstance(self._redis, RedisCluster) and not self._is_cluster_healthy():
            logger.warning(
                "Detected unhealthy RedisCluster client during put(); attempting recovery with standalone client."
            )
            self._attempt_cluster_recovery()

        # Store checkpoint and blob data based on cluster mode
        blob_keys = []
        
        # Enhanced cluster mode detection - check if we're actually on a cluster
        actual_cluster_mode = self.cluster_mode and self._is_cluster_healthy()
        if not actual_cluster_mode:
            # Optionally try to detect cluster if not already enabled
            try:
                if hasattr(self._redis, 'cluster'):
                    self._redis.cluster('info')
                    if self._is_cluster_healthy():
                        actual_cluster_mode = True
                        logger.info("Detected healthy Redis cluster via cluster info command, enabling cluster mode")
            except Exception:
                pass
        
        if actual_cluster_mode:
            # For cluster mode, handle operations individually
            self._redis.hset(checkpoint_key, mapping=cleaned_checkpoint_data)
            
            # Store blob values individually
            blobs = self._dump_blobs(
                storage_safe_thread_id,
                storage_safe_checkpoint_ns,
                copy.get("channel_values", {}),
                new_versions,
            )

            for blob_key, blob_data in blobs:
                # Clean blob_data to ensure no None values
                cleaned_blob_data = {}
                for k, v in blob_data.items():
                    if v is not None:
                        cleaned_blob_data[k] = v
                    else:
                        # Convert None to empty string for Redis compatibility
                        cleaned_blob_data[k] = ""
                
                # Store blob as hash
                self._redis.hset(blob_key, mapping=cleaned_blob_data)
                blob_keys.append(blob_key)
        else:
            # For non-cluster mode, use pipeline for efficiency
            try:
                pipeline = self._redis.pipeline()
                
                # Store checkpoint data as hash
                pipeline.hset(checkpoint_key, mapping=cleaned_checkpoint_data)

                # Store blob values
                blobs = self._dump_blobs(
                    storage_safe_thread_id,
                    storage_safe_checkpoint_ns,
                    copy.get("channel_values", {}),
                    new_versions,
                )

                for blob_key, blob_data in blobs:
                    # Clean blob_data to ensure no None values
                    cleaned_blob_data = {}
                    for k, v in blob_data.items():
                        if v is not None:
                            cleaned_blob_data[k] = v
                        else:
                            # Convert None to empty string for Redis compatibility
                            cleaned_blob_data[k] = ""
                    
                    # Store blob as hash
                    pipeline.hset(blob_key, mapping=cleaned_blob_data)
                    blob_keys.append(blob_key)
                
                pipeline.execute()
                
            except Exception as e:
                logger.warning(f"Pipeline execution failed: {e}. Falling back to individual operations.")
                
                # Detect Redis cluster redirection errors (MOVED / ASK). These
                # indicate that we're talking to a cluster node with a non-cluster
                # client.  Attempt to rebuild a proper RedisCluster client and
                # retry the write once.
                err_msg = str(e)
                if "MOVED" in err_msg or "ASK" in err_msg:
                    logger.warning(
                        f"Redis redirection error detected ({err_msg}). Initiating cluster recovery and retry."
                    )

                    # Force cluster mode so subsequent logic follows the cluster path
                    self.cluster_mode = True
                    try:
                        self._attempt_cluster_recovery()
                        # Retry the put once with a healthy cluster client
                        return self.put(config, checkpoint, metadata, new_versions)
                    except Exception as recovery_exc:
                        logger.error(
                            f"Cluster recovery failed after redirection error: {recovery_exc}"
                        )
                        raise recovery_exc

                # Fallback to individual operations if pipeline fails
                try:
                    # Store checkpoint data as hash
                    self._redis.hset(checkpoint_key, mapping=cleaned_checkpoint_data)

                    # Store blob values individually
                    blobs = self._dump_blobs(
                        storage_safe_thread_id,
                        storage_safe_checkpoint_ns,
                        copy.get("channel_values", {}),
                        new_versions,
                    )

                    for blob_key, blob_data in blobs:
                        # Clean blob_data to ensure no None values
                        cleaned_blob_data = {}
                        for k, v in blob_data.items():
                            if v is not None:
                                cleaned_blob_data[k] = v
                            else:
                                # Convert None to empty string for Redis compatibility
                                cleaned_blob_data[k] = ""
                        
                        # Store blob as hash
                        self._redis.hset(blob_key, mapping=cleaned_blob_data)
                        blob_keys.append(blob_key)
                        
                except Exception as e2:
                    # Detect Redis cluster redirection errors (MOVED / ASK). These
                    # indicate that we're talking to a cluster node with a non-cluster
                    # client.  Attempt to rebuild a proper RedisCluster client and
                    # retry the write once.
                    err_msg = str(e2)
                    if "MOVED" in err_msg or "ASK" in err_msg:
                        logger.warning(
                            f"Redis redirection error detected ({err_msg}). Initiating cluster recovery and retry."
                        )

                        # Force cluster mode so subsequent logic follows the cluster path
                        self.cluster_mode = True
                        try:
                            self._attempt_cluster_recovery()
                            # Retry the put once with a healthy cluster client
                            return self.put(config, checkpoint, metadata, new_versions)
                        except Exception as recovery_exc:
                            logger.error(
                                f"Cluster recovery failed after redirection error: {recovery_exc}"
                            )
                            raise recovery_exc

                    logger.error(f"Both pipeline and individual operations failed: {e2}")
                    raise

        # Apply TTL to checkpoint and blob keys if configured
        if self.ttl_config and "default_ttl" in self.ttl_config:
            self._apply_ttl_to_keys(checkpoint_key, blob_keys)

        return next_config

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis using basic operations."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        if checkpoint_id and checkpoint_id != EMPTY_ID_SENTINEL:
            # Get specific checkpoint
            checkpoint_key = BaseRedisSaver._make_redis_checkpoint_key(
                to_storage_safe_id(thread_id),
                to_storage_safe_str(checkpoint_ns),
                to_storage_safe_id(checkpoint_id),
            )
            
            checkpoint_data = self._redis.hgetall(checkpoint_key)
            if not checkpoint_data:
                return None
                
        else:
            # Get latest checkpoint for thread using cluster-safe scanning
            pattern = f"checkpoint:{to_storage_safe_id(thread_id)}:{to_storage_safe_str(checkpoint_ns)}:*"
            
            keys = self._scan_keys_cluster_safe(pattern, count=1000)
            # Convert to bytes for consistency
            keys = [k.encode() if isinstance(k, str) else k for k in keys]
            
            if not keys:
                return None
                
            # Sort keys to get the latest (assuming checkpoint_id is sortable)
            keys.sort(reverse=True)
            checkpoint_key = keys[0].decode() if isinstance(keys[0], bytes) else keys[0]
            
            checkpoint_data = self._redis.hgetall(checkpoint_key)
            if not checkpoint_data:
                return None

        # Decode checkpoint data
        decoded_data = {}
        for k, v in checkpoint_data.items():
            key_str = k.decode() if isinstance(k, bytes) else k
            val_str = v.decode() if isinstance(v, bytes) else v
            decoded_data[key_str] = val_str

        doc_thread_id = from_storage_safe_id(decoded_data["thread_id"])
        doc_checkpoint_ns = from_storage_safe_str(decoded_data["checkpoint_ns"])
        doc_checkpoint_id = from_storage_safe_id(decoded_data["checkpoint_id"])
        doc_parent_checkpoint_id = from_storage_safe_id(decoded_data.get("parent_checkpoint_id", ""))

        # If refresh_on_read is enabled, refresh TTL
        if self.ttl_config and self.ttl_config.get("refresh_on_read"):
            # Get related blob and write keys using cluster-safe scanning
            blob_pattern = f"checkpoint_blob:{to_storage_safe_id(doc_thread_id)}:{to_storage_safe_str(doc_checkpoint_ns)}:*"
            write_pattern = f"checkpoint_write:{to_storage_safe_id(doc_thread_id)}:{to_storage_safe_str(doc_checkpoint_ns)}:{to_storage_safe_id(doc_checkpoint_id)}:*"
            
            all_keys = [checkpoint_key]
            
            # Get blob keys using cluster-safe scanning
            blob_keys = self._scan_keys_cluster_safe(blob_pattern, count=1000)
            all_keys.extend(blob_keys)
                    
            # Get write keys using cluster-safe scanning
            write_keys = self._scan_keys_cluster_safe(write_pattern, count=1000)
            all_keys.extend(write_keys)

            # Apply TTL to all related keys
            if len(all_keys) > 1:
                self._apply_ttl_to_keys(all_keys[0], all_keys[1:])

        # Fetch channel_values
        channel_values = self.get_channel_values(
            thread_id=doc_thread_id,
            checkpoint_ns=doc_checkpoint_ns,
            checkpoint_id=doc_checkpoint_id,
        )

        # Fetch pending_sends from parent checkpoint
        pending_sends = []
        if doc_parent_checkpoint_id:
            pending_sends = self._load_pending_sends(
                thread_id=doc_thread_id,
                checkpoint_ns=doc_checkpoint_ns,
                parent_checkpoint_id=doc_parent_checkpoint_id,
            )

        # Parse metadata
        metadata_str = decoded_data.get("metadata", "{}")
        try:
            metadata_dict = json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            metadata_dict = {}

        # Sanitize metadata
        sanitized_metadata = {
            k.replace("\u0000", ""): (
                v.replace("\u0000", "") if isinstance(v, str) else v
            )
            for k, v in metadata_dict.items()
        }
        metadata = cast(CheckpointMetadata, sanitized_metadata)

        config_param: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": doc_checkpoint_id,
            }
        }

        checkpoint_param = self._load_checkpoint(
            decoded_data.get("checkpoint", "{}"),
            channel_values,
            pending_sends,
        )

        pending_writes = self._load_pending_writes(
            thread_id, checkpoint_ns, doc_checkpoint_id
        )

        return CheckpointTuple(
            config=config_param,
            checkpoint=checkpoint_param,
            metadata=metadata,
            parent_config=None,
            pending_writes=pending_writes,
        )

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Union[Redis, RedisCluster]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        ttl: Optional[Dict[str, Any]] = None,
    ) -> Iterator["RedisSaver"]:
        """Create a new RedisSaver instance."""
        saver: Optional["RedisSaver"] = None
        try:
            saver = cls(
                redis_url=redis_url,
                redis_client=redis_client,
                connection_args=connection_args,
                ttl=ttl,
            )

            yield saver
        finally:
            if saver and saver._owns_its_client:  # Ensure saver is not None
                saver._redis.close()
                if hasattr(saver._redis, 'connection_pool'):
                    saver._redis.connection_pool.disconnect()

    def get_channel_values(
        self, thread_id: str, checkpoint_ns: str = "", checkpoint_id: str = ""
    ) -> Dict[str, Any]:
        """Retrieve channel_values using basic Redis operations."""
        storage_safe_thread_id = to_storage_safe_id(thread_id)
        storage_safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        storage_safe_checkpoint_id = to_storage_safe_id(checkpoint_id)

        # Get checkpoint to find channel_versions
        checkpoint_key = BaseRedisSaver._make_redis_checkpoint_key(
            storage_safe_thread_id,
            storage_safe_checkpoint_ns,
            storage_safe_checkpoint_id,
        )

        checkpoint_data = self._redis.hgetall(checkpoint_key)
        if not checkpoint_data:
            return {}

        # Parse checkpoint data to get channel_versions
        checkpoint_str = checkpoint_data.get(b"checkpoint", b"{}").decode()
        try:
            checkpoint_dict = json.loads(checkpoint_str)
            channel_versions = checkpoint_dict.get("channel_versions", {})
        except (json.JSONDecodeError, TypeError):
            return {}

        if not channel_versions:
            return {}

        channel_values = {}
        for channel, version in channel_versions.items():
            blob_key = BaseRedisSaver._make_redis_checkpoint_blob_key(
                storage_safe_thread_id,
                storage_safe_checkpoint_ns,
                channel,
                str(version),
            )

            blob_data = self._redis.hgetall(blob_key)
            if blob_data:
                blob_type = blob_data.get(b"type", b"").decode()
                blob_content = blob_data.get(b"blob")

                if blob_content and blob_type and blob_type != "empty":
                    try:
                        channel_values[channel] = self.serde.loads_typed(
                            (blob_type, blob_content)
                        )
                    except Exception as e:
                        logger.warning(f"Error loading blob for channel {channel}: {e}")

        return channel_values

    def _load_pending_sends(
        self,
        thread_id: str,
        checkpoint_ns: str,
        parent_checkpoint_id: str,
    ) -> List[Tuple[str, bytes]]:
        """Load pending sends for a parent checkpoint using basic Redis operations."""
        # Find write keys for parent checkpoint with TASKS channel using cluster-safe scanning
        pattern = f"checkpoint_write:{to_storage_safe_id(thread_id)}:{to_storage_safe_str(checkpoint_ns)}:{to_storage_safe_id(parent_checkpoint_id)}:*"
        
        keys = self._scan_keys_cluster_safe(pattern, count=1000)
        # Convert to bytes for consistency
        keys = [k.encode() if isinstance(k, str) else k for k in keys]

        # Filter for TASKS channel and collect writes
        writes = []
        for key_bytes in keys:
            key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
            write_data = self._redis.hgetall(key)
            
            if write_data:
                channel = write_data.get(b"channel", b"").decode()
                if channel == TASKS:
                    write_type = write_data.get(b"type", b"").decode()
                    write_blob = write_data.get(b"blob", b"")
                    task_path = write_data.get(b"task_path", b"").decode()
                    task_id = write_data.get(b"task_id", b"").decode()
                    idx = int(write_data.get(b"idx", b"0"))
                    
                    writes.append((task_path, task_id, idx, write_type, write_blob))

        # Sort by task_path, task_id, idx
        writes.sort(key=lambda x: (x[0], x[1], x[2]))

        # Return type and blob pairs
        return [(write[3], write[4]) for write in writes]

    def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[Any]:
        """Load pending writes using basic Redis operations."""
        if checkpoint_id is None:
            return []

        # Find write keys for this checkpoint using cluster-safe scanning
        pattern = f"checkpoint_write:{to_storage_safe_id(thread_id)}:{to_storage_safe_str(checkpoint_ns)}:{to_storage_safe_id(checkpoint_id)}:*"
        
        keys = self._scan_keys_cluster_safe(pattern, count=1000)
        # Convert to bytes for consistency
        keys = [k.encode() if isinstance(k, str) else k for k in keys]

        # Collect writes data
        writes_dict = {}
        for key_bytes in keys:
            key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
            write_data = self._redis.hgetall(key)
            
            if write_data:
                task_id = write_data.get(b"task_id", b"").decode()
                idx = write_data.get(b"idx", b"0").decode()
                channel = write_data.get(b"channel", b"").decode()
                write_type = write_data.get(b"type", b"").decode()
                write_blob = write_data.get(b"blob", b"")
                
                writes_dict[(task_id, idx)] = {
                    "task_id": task_id,
                    "idx": idx,
                    "channel": channel,
                    "type": write_type,
                    "blob": write_blob,
                }

        return BaseRedisSaver._load_writes(self.serde, writes_dict)

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a specific thread ID."""
        storage_safe_thread_id = to_storage_safe_id(thread_id)

        # Collect all keys to delete using cluster-safe scanning
        patterns = [
            f"checkpoint:{storage_safe_thread_id}:*",
            f"checkpoint_blob:{storage_safe_thread_id}:*", 
            f"checkpoint_write:{storage_safe_thread_id}:*"
        ]
        
        keys_to_delete = []
        for pattern in patterns:
            pattern_keys = self._scan_keys_cluster_safe(pattern, count=1000)
            keys_to_delete.extend(pattern_keys)

        # Execute deletions based on cluster mode
        if keys_to_delete:
            # Validate cluster mode before using cluster operations
            use_cluster_mode = self.cluster_mode
            if use_cluster_mode:
                try:
                    # Validate cluster client is working before using individual operations
                    if hasattr(self._redis, 'get_default_node'):
                        default_node = self._redis.get_default_node()
                        if default_node is None or not hasattr(default_node, 'redis_connection'):
                            logger.warning("Cluster validation failed in delete_thread. Using pipeline mode.")
                            use_cluster_mode = False
                except Exception as e:
                    logger.warning(f"Cluster validation failed in delete_thread: {e}. Using pipeline mode.")
                    use_cluster_mode = False
            
            if use_cluster_mode:
                # For cluster mode, delete keys individually
                for key in keys_to_delete:
                    self._redis.delete(key)
            else:
                # For non-cluster mode, use pipeline for efficiency
                try:
                    pipeline = self._redis.pipeline()
                    for key in keys_to_delete:
                        pipeline.delete(key)
                    pipeline.execute()
                except Exception as e:
                    logger.warning(f"Pipeline deletion failed: {e}. Falling back to individual deletions.")
                    # Fallback to individual deletions
                    for key in keys_to_delete:
                        try:
                            self._redis.delete(key)
                        except Exception as key_error:
                            logger.warning(f"Failed to delete key {key}: {key_error}")

    def _apply_ttl_to_keys(
        self,
        main_key: str,
        related_keys: Optional[List[str]] = None,
        ttl_minutes: Optional[float] = None,
    ) -> Any:
        """Apply Redis native TTL to keys.

        Args:
            main_key: The primary Redis key
            related_keys: Additional Redis keys that should expire at the same time
            ttl_minutes: Time-to-live in minutes, overrides default_ttl if provided

        Returns:
            Result of the Redis operation
        """
        if ttl_minutes is None:
            # Check if there's a default TTL in config
            if self.ttl_config and "default_ttl" in self.ttl_config:
                ttl_minutes = self.ttl_config.get("default_ttl")

        if ttl_minutes is not None:
            ttl_seconds = int(ttl_minutes * 60)

            # Validate cluster mode before using cluster operations
            use_cluster_mode = self.cluster_mode
            if use_cluster_mode:
                try:
                    # Validate cluster client is working before using individual operations
                    if hasattr(self._redis, 'get_default_node'):
                        default_node = self._redis.get_default_node()
                        if default_node is None or not hasattr(default_node, 'redis_connection'):
                            logger.warning("Cluster validation failed in _apply_ttl_to_keys. Using pipeline mode.")
                            use_cluster_mode = False
                except Exception as e:
                    logger.warning(f"Cluster validation failed in _apply_ttl_to_keys: {e}. Using pipeline mode.")
                    use_cluster_mode = False

            if use_cluster_mode:
                # For cluster mode, execute TTL operations individually
                self._redis.expire(main_key, ttl_seconds)

                if related_keys:
                    for key in related_keys:
                        self._redis.expire(key, ttl_seconds)

                return True
            else:
                # For non-cluster mode, use pipeline for efficiency
                try:
                    pipeline = self._redis.pipeline()

                    # Set TTL for main key
                    pipeline.expire(main_key, ttl_seconds)

                    # Set TTL for related keys
                    if related_keys:
                        for key in related_keys:
                            pipeline.expire(key, ttl_seconds)

                    return pipeline.execute()
                except Exception as e:
                    logger.warning(f"Pipeline TTL setting failed: {e}. Falling back to individual TTL operations.")
                    # Fallback to individual TTL operations
                    try:
                        self._redis.expire(main_key, ttl_seconds)
                        if related_keys:
                            for key in related_keys:
                                try:
                                    self._redis.expire(key, ttl_seconds)
                                except Exception as key_error:
                                    logger.warning(f"Failed to set TTL for key {key}: {key_error}")
                        return True
                    except Exception as e2:
                        logger.error(f"Both pipeline and individual TTL operations failed: {e2}")
                        return False

        return None

    def _scan_keys_cluster_safe(self, pattern: str, count: int = 1000) -> List[str]:
        """Scan for keys in a cluster-safe way."""
        if not self.cluster_mode:
            # Use regular SCAN for non-cluster mode
            keys = []
            cursor = 0
            while True:
                cursor, batch_keys = self._redis.scan(cursor, match=pattern, count=count)
                keys.extend([k.decode() if isinstance(k, bytes) else k for k in batch_keys])
                if cursor == 0:
                    break
            return keys
        else:
            # For cluster mode, use a simpler approach that works with redis-py cluster client
            all_keys: List[str] = []
            try:
                # Method 1: Try using the cluster client's built-in scan_iter
                if hasattr(self._redis, 'scan_iter'):
                    for key in self._redis.scan_iter(match=pattern, count=count):
                        key_str = key.decode() if isinstance(key, bytes) else key
                        all_keys.append(key_str)
                    return all_keys
                
                # Method 2: Try direct SCAN on cluster (redis-py handles routing)
                cursor = 0
                while True:
                    cursor, batch_keys = self._redis.scan(cursor, match=pattern, count=count)
                    all_keys.extend([k.decode() if isinstance(k, bytes) else k for k in batch_keys])
                    if cursor == 0:
                        break
                return all_keys
                
            except Exception as e:
                logger.warning(f"Cluster SCAN failed, trying KEYS fallback: {e}")
                try:
                    # Fallback: use KEYS command (less efficient but works)
                    keys = self._redis.keys(pattern)
                    all_keys = [k.decode() if isinstance(k, bytes) else k for k in keys]
                    return all_keys
                except Exception as e2:
                    logger.warning(f"KEYS fallback failed: {e2}. Trying per-startup-node scan")

                    # Final fallback: iterate over original startup nodes with standalone connections
                    aggregated: list[str] = []
                    for node in getattr(self, "_startup_nodes", []):
                        try:
                            host = node.get("host")
                            port = node.get("port")
                            pwd = node.get("password")
                            if not host or not port:
                                continue
                            
                            # Build connection args including SSL settings
                            standalone_args = {
                                'host': host,
                                'port': port,
                                'socket_timeout': 2
                            }
                            if pwd:
                                standalone_args['password'] = pwd
                            
                            # Add SSL configuration if present in node config
                            ssl_keys = ['ssl', 'ssl_cert_reqs', 'ssl_ca_certs', 'ssl_certfile', 'ssl_keyfile']
                            for ssl_key in ssl_keys:
                                if ssl_key in node:
                                    standalone_args[ssl_key] = node[ssl_key]
                            
                            standalone_client = Redis(**standalone_args)
                            cursor_inner = 0
                            while True:
                                cursor_inner, bkeys = standalone_client.scan(cursor_inner, match=pattern, count=count)
                                aggregated.extend([k.decode() if isinstance(k, bytes) else k for k in bkeys])
                                if cursor_inner == 0:
                                    break
                            standalone_client.close()
                        except Exception as node_exc:
                            logger.debug(f"Node scan error on {node}: {node_exc}")
                            continue
                    return aggregated

    def _is_cluster_healthy(self) -> bool:
        """Check if the current RedisCluster client appears usable."""
        if not isinstance(self._redis, RedisCluster):
            return False
        try:
            if hasattr(self._redis, 'get_default_node'):
                node = self._redis.get_default_node()
                if node is None:
                    return False
                if getattr(node, 'redis_connection', None) is None:
                    return False
            # simple ping
            self._redis.ping()
            return True
        except Exception:
            return False


__all__ = [
    "__version__",
    "__lib_name__",
    "RedisSaver",
    "AsyncRedisSaver",
    "BaseRedisSaver",
    "ShallowRedisSaver",
    "AsyncShallowRedisSaver",
]
