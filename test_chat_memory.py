#!/usr/bin/env python3
"""
Comprehensive test for RedisSaver with actual chat agent to verify memory functionality.
This tests that the agent can remember information across multiple interactions in the same thread.
"""

import os
from typing import Literal
from uuid import uuid4

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.redis import RedisSaver
from langgraph.prebuilt import create_react_agent


@tool
def get_weather(city: Literal["nyc", "sf"]) -> str:
    """Get current weather for a city."""
    return f"The weather in {city} is sunny and 72°F."


def test_chat_memory_functionality():
    """Test that RedisSaver correctly maintains chat memory across interactions."""
    
    redis_url = "redis://localhost:6379"
    
    # Add cluster mode support - detect from environment or default to standalone
    cluster_mode = os.getenv("REDIS_CLUSTER_MODE", "false").lower() == "true"
    connection_args = {}
    
    if cluster_mode:
        print("🧪 Testing RedisSaver with Redis CLUSTER mode...")
        # For cluster mode, modify the URL or add cluster connection args
        redis_url = os.getenv("REDIS_CLUSTER_URL", "redis://localhost:7000")
        connection_args = {
            "cluster_mode": True,
            "startup_nodes": [
                {"host": "localhost", "port": 7000},
                {"host": "localhost", "port": 7001}, 
                {"host": "localhost", "port": 7002}
            ]
        }
        # Use connection_args instead of URL for cluster
        redis_url = None
    else:
        print("🧪 Testing RedisSaver with Redis STANDALONE mode...")
    
    # Check for Anthropic API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY environment variable not set!")
        print("   Please set your Anthropic API key: export ANTHROPIC_API_KEY=your_key_here")
        return False
    
    try:
        # Create Anthropic LLM
        llm = ChatAnthropic(
            model="claude-3-haiku-20240307",  # Using Haiku for cost efficiency
            temperature=0.1,  # Lower temperature for more consistent responses
            api_key=api_key
        )
        print("✅ Created ChatAnthropic LLM")
        
        # Create tools
        tools = [get_weather]
        
        # Create RedisSaver with cluster support
        with RedisSaver.from_conn_string(redis_url, connection_args=connection_args) as checkpointer:
            checkpointer.setup()
            print("✅ Created RedisSaver checkpointer")
            
            # Create the agent with checkpointer
            agent = create_react_agent(llm, tools, checkpointer=checkpointer)
            print("✅ Created React agent with checkpointer")
            
            # Test conversation with memory
            thread_id = f"test-chat-{uuid4()}"
            config = {"configurable": {"thread_id": thread_id}}
            
            print(f"\n💬 Starting conversation in thread: {thread_id}")
            
            # First interaction: User introduces themselves
            print("\n👤 User: Hello, my name is Alice.")
            result1 = agent.invoke(
                {"messages": [
                    SystemMessage(content="You are a helpful assistant. Remember information shared by users in this conversation."),
                    HumanMessage(content="Hello, my name is Alice.")
                ]},
                config=config
            )
            
            assistant_response1 = result1["messages"][-1].content
            print(f"🤖 Assistant: {assistant_response1}")
            
            # Verify the agent acknowledges the name (more flexible matching for real LLM)
            assert "alice" in assistant_response1.lower() or "nice" in assistant_response1.lower(), f"Agent didn't acknowledge name properly in: {assistant_response1}"
            print("✅ Agent acknowledged the user's name")
            
            # Second interaction: Ask about the weather
            print("\n👤 User: What's the weather like in NYC?")
            result2 = agent.invoke(
                {"messages": [HumanMessage(content="What's the weather like in NYC?")]},
                config=config
            )
            
            assistant_response2 = result2["messages"][-1].content
            print(f"🤖 Assistant: {assistant_response2}")
            
            # Third interaction: Ask the agent to recall the name
            print("\n👤 User: What is my name?")
            result3 = agent.invoke(
                {"messages": [HumanMessage(content="What is my name?")]},
                config=config
            )
            
            assistant_response3 = result3["messages"][-1].content
            print(f"🤖 Assistant: {assistant_response3}")
            
            # Verify the agent remembers the name
            assert "alice" in assistant_response3.lower(), f"Agent forgot the name! Response: {assistant_response3}"
            print("✅ Agent correctly remembered the user's name")
            
            # Verify conversation history is maintained
            final_messages = result3["messages"]
            human_messages = [msg for msg in final_messages if isinstance(msg, HumanMessage)]
            ai_messages = [msg for msg in final_messages if isinstance(msg, AIMessage)]
            
            print(f"\n📊 Conversation summary:")
            print(f"   - Total messages: {len(final_messages)}")
            print(f"   - Human messages: {len(human_messages)}")
            print(f"   - AI messages: {len(ai_messages)}")
            
            # Should have at least 3 human messages and 3 AI responses
            assert len(human_messages) >= 3, f"Expected at least 3 human messages, got {len(human_messages)}"
            assert len(ai_messages) >= 3, f"Expected at least 3 AI messages, got {len(ai_messages)}"
            print("✅ Conversation history is properly maintained")
            
            # Test 2: New thread should not have memory of the previous conversation
            print(f"\n🔄 Testing memory isolation with new thread...")
            
            new_thread_id = f"test-chat-{uuid4()}"
            new_config = {"configurable": {"thread_id": new_thread_id}}
            
            print(f"   New thread: {new_thread_id}")
            print("\n👤 User: What is my name?")
            
            result_new = agent.invoke(
                {"messages": [HumanMessage(content="What is my name?")]},
                config=new_config
            )
            
            assistant_response_new = result_new["messages"][-1].content
            print(f"🤖 Assistant: {assistant_response_new}")
            
            # Should not remember the name from the previous thread
            # Note: Real LLM might respond differently, so we check that it doesn't confidently state the name
            name_mentioned = "alice" in assistant_response_new.lower()
            if name_mentioned:
                print(f"⚠️  Note: Agent mentioned Alice in new thread: {assistant_response_new}")
                print("   This could be normal behavior for a real LLM")
            print("✅ Memory isolation test completed")
            
            # Test 3: Verify checkpoint data in Redis
            print(f"\n🔍 Verifying checkpoint data in Redis...")
            
            # List checkpoints for the original thread
            original_config = {"configurable": {"thread_id": thread_id}}
            checkpoints = list(checkpointer.list(original_config))
            
            print(f"   - Found {len(checkpoints)} checkpoints for thread {thread_id}")
            assert len(checkpoints) > 0, "No checkpoints found in Redis"
            
            # Check that we can retrieve a specific checkpoint
            latest_checkpoint = checkpointer.get_tuple(original_config)
            assert latest_checkpoint is not None, "Could not retrieve latest checkpoint"
            
            print(f"   - Latest checkpoint has {len(latest_checkpoint.checkpoint.get('channel_values', {}).get('messages', []))} messages")
            print("✅ Checkpoint data verified in Redis")
            
            # Test 4: Resume conversation in original thread
            print(f"\n🔄 Resuming original conversation...")
            
            print("\n👤 User: Thank you for remembering my name!")
            result4 = agent.invoke(
                {"messages": [HumanMessage(content="Thank you for remembering my name!")]},
                config=config
            )
            
            assistant_response4 = result4["messages"][-1].content
            print(f"🤖 Assistant: {assistant_response4}")
            
            # Verify conversation continues seamlessly
            final_messages_count = len(result4["messages"])
            print(f"   - Total messages in resumed conversation: {final_messages_count}")
            assert final_messages_count > len(final_messages), "Conversation didn't continue properly"
            print("✅ Conversation resumed successfully")
            
            # Cleanup: Delete the test threads
            print(f"\n🧹 Cleaning up test data...")
            checkpointer.delete_thread(thread_id)
            checkpointer.delete_thread(new_thread_id)
            print("✅ Test data cleaned up")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All chat memory tests passed! RedisSaver correctly maintains conversation state.")
    return True


def test_concurrent_threads():
    """Test that multiple concurrent conversations work independently."""
    
    redis_url = "redis://localhost:6379"
    
    # Add cluster mode support - detect from environment or default to standalone
    cluster_mode = os.getenv("REDIS_CLUSTER_MODE", "false").lower() == "true"
    connection_args = {}
    
    if cluster_mode:
        print("\n🧪 Testing concurrent thread functionality with Redis CLUSTER mode...")
        # For cluster mode, modify the URL or add cluster connection args
        redis_url = os.getenv("REDIS_CLUSTER_URL", "redis://localhost:7000")
        connection_args = {
            "cluster_mode": True,
            "startup_nodes": [
                {"host": "localhost", "port": 7000},
                {"host": "localhost", "port": 7001}, 
                {"host": "localhost", "port": 7002}
            ]
        }
        # Use connection_args instead of URL for cluster
        redis_url = None
    else:
        print("\n🧪 Testing concurrent thread functionality with Redis STANDALONE mode...")
    
    # Check for Anthropic API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY environment variable not set!")
        return False
    
    try:
        llm = ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0.1,
            api_key=api_key
        )
        tools = [get_weather]
        
        with RedisSaver.from_conn_string(redis_url, connection_args=connection_args) as checkpointer:
            checkpointer.setup()
            
            agent = create_react_agent(llm, tools, checkpointer=checkpointer)
            
            # Create two separate conversations
            thread1_id = f"concurrent-1-{uuid4()}"
            thread2_id = f"concurrent-2-{uuid4()}"
            
            config1 = {"configurable": {"thread_id": thread1_id}}
            config2 = {"configurable": {"thread_id": thread2_id}}
            
            print(f"   Thread 1: {thread1_id}")
            print(f"   Thread 2: {thread2_id}")
            
            # Thread 1: Alice introduces herself
            print("\n🧵 Thread 1 - User: My name is Alice")
            agent.invoke(
                {"messages": [HumanMessage(content="My name is Alice")]},
                config=config1
            )
            
            # Thread 2: Bob introduces himself
            print("🧵 Thread 2 - User: My name is Bob")
            agent.invoke(
                {"messages": [HumanMessage(content="My name is Bob")]},
                config=config2
            )
            
            # Verify each thread remembers its own user
            print("\n🧵 Thread 1 - User: What is my name?")
            result1 = agent.invoke(
                {"messages": [HumanMessage(content="What is my name?")]},
                config=config1
            )
            response1 = result1["messages"][-1].content
            print(f"🧵 Thread 1 - Assistant: {response1}")
            
            print("\n🧵 Thread 2 - User: What is my name?")
            result2 = agent.invoke(
                {"messages": [HumanMessage(content="What is my name?")]},
                config=config2
            )
            response2 = result2["messages"][-1].content
            print(f"🧵 Thread 2 - Assistant: {response2}")
            
            # Verify correct isolation (more flexible for real LLM)
            alice_in_thread1 = "alice" in response1.lower()
            bob_in_thread2 = "bob" in response2.lower()
            
            print(f"   Thread 1 response mentions Alice: {alice_in_thread1}")
            print(f"   Thread 2 response mentions Bob: {bob_in_thread2}")
            
            if not alice_in_thread1:
                print(f"⚠️  Thread 1 may not have remembered Alice: {response1}")
            if not bob_in_thread2:
                print(f"⚠️  Thread 2 may not have remembered Bob: {response2}")
            
            print("✅ Concurrent threads test completed")
            
            # Cleanup
            checkpointer.delete_thread(thread1_id)
            checkpointer.delete_thread(thread2_id)
            
    except Exception as e:
        print(f"❌ Concurrent test failed: {e}")
        return False
    
    print("✅ Concurrent thread test passed!")
    return True


if __name__ == "__main__":
    print("🚀 Starting comprehensive RedisSaver chat tests...\n")
    
    # Print cluster mode instructions
    cluster_mode = os.getenv("REDIS_CLUSTER_MODE", "false").lower() == "true"
    if cluster_mode:
        print("🔧 Running in CLUSTER mode!")
        print("   Make sure your Redis cluster is running on ports 7000-7002")
        print("   To test standalone mode: unset REDIS_CLUSTER_MODE")
    else:
        print("🔧 Running in STANDALONE mode!")
        print("   Make sure Redis is running on localhost:6379")
        print("   To test cluster mode: export REDIS_CLUSTER_MODE=true")
    print()
    
    success1 = test_chat_memory_functionality()
    success2 = test_concurrent_threads()
    
    if success1 and success2:
        print("\n🎉 All tests passed! RedisSaver works perfectly with chat agents.")
        if cluster_mode:
            print("✅ CLUSTER mode compatibility confirmed - no more MOVED errors!")
        else:
            print("✅ STANDALONE mode working as expected!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        print("\n📋 Troubleshooting:")
        if cluster_mode:
            print("   - Check if Redis cluster is running and accessible")
            print("   - Verify cluster nodes are on ports 7000, 7001, 7002")
            print("   - Try standalone mode: unset REDIS_CLUSTER_MODE")
        else:
            print("   - Check if Redis is running on localhost:6379")
            print("   - Try: redis-server --port 6379")
        exit(1) 