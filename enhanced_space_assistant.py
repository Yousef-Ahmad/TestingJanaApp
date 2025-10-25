#!/usr/bin/env python3
"""
Enhanced Space Science RAG Assistant with GPT-4 Integration
Handles all errors gracefully - won't crash the app.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

# Try importing dependencies - make everything optional
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not available")

try:
    from query_optimizer import SpaceScienceQueryOptimizer
    QUERY_OPTIMIZER_AVAILABLE = True
except ImportError:
    QUERY_OPTIMIZER_AVAILABLE = False
    print("⚠️  query_optimizer not available")
    
    # Fallback query optimizer
    class SpaceScienceQueryOptimizer:
        def optimize_query(self, query: str) -> Dict[str, Any]:
            return {
                'optimized_query': query,
                'original_query': query,
                'extracted_keywords': [],
                'relevant_topics': []
            }

class EnhancedSpaceScienceAssistant:
    """Enhanced RAG Assistant - works even with missing dependencies."""
    
    def __init__(self, 
                 knowledge_base_path: str = "space_science_knowledge_base.json",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 openai_api_key: Optional[str] = None,
                 chroma_persist_dir: str = "./enhanced_chroma_db"):
        """Initialize the assistant."""
        print("Initializing Enhanced Space Science Assistant...")
        
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model_name = embedding_model
        self.chroma_persist_dir = chroma_persist_dir
        
        # Initialize OpenAI
        self.openai_client = self._initialize_openai(openai_api_key)
        
        # Initialize components
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.query_optimizer = SpaceScienceQueryOptimizer()
        self.conversation_history = []
        self.knowledge_base = []
        
        # Try to initialize (but don't crash if it fails)
        try:
            self._safe_initialize()
        except Exception as e:
            print(f"⚠️  Initialization warning: {e}")
            print("   Assistant will work in limited mode")
    
    def _safe_initialize(self):
        """Safely initialize components."""
        # Load embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                print("✅ Embedding model loaded")
            except Exception as e:
                print(f"⚠️  Could not load embedding model: {e}")
        
        # Initialize ChromaDB
        if CHROMADB_AVAILABLE:
            try:
                print("Setting up ChromaDB...")
                self.chroma_client = chromadb.PersistentClient(
                    path=self.chroma_persist_dir,
                    settings=Settings(anonymized_telemetry=False)
                )
                self.collection_name = "enhanced_space_science_kb"
                self._initialize_vector_db()
                print("✅ ChromaDB initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize ChromaDB: {e}")
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base(self.knowledge_base_path)
        print(f"✅ System initialized with {len(self.knowledge_base)} knowledge chunks")
    
    def initialize(self):
        """Compatibility method."""
        print("Enhanced assistant initialization complete.")
        return True
    
    def _initialize_openai(self, api_key: Optional[str]):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            print("⚠️  OpenAI library not available")
            return None
            
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            print("⚠️  No OpenAI API key - basic mode only")
            return None
        
        try:
            client = openai.OpenAI()
            client.models.list()  # Test connection
            print("✅ OpenAI GPT-4 connection established")
            return client
        except Exception as e:
            print(f"⚠️  OpenAI connection failed: {e}")
            return None
    
    def _load_knowledge_base(self, file_path: str) -> List[Dict]:
        """Load knowledge base from JSON file."""
        try:
            if not os.path.exists(file_path):
                print(f"⚠️  Knowledge base file not found: {file_path}")
                return []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('knowledge_base', {}).get('chunks', [])
        except Exception as e:
            print(f"⚠️  Error loading knowledge base: {e}")
            return []
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB collection."""
        if not self.chroma_client:
            return
            
        try:
            # Try to get existing collection
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection with {self.collection.count()} documents")
        except:
            # Create new collection
            print("Creating new collection...")
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Enhanced Space Science KB"}
            )
            if self.knowledge_base and self.embedding_model:
                self._populate_vector_db()
    
    def _populate_vector_db(self):
        """Populate ChromaDB with knowledge chunks."""
        if not self.knowledge_base or not self.collection or not self.embedding_model:
            return
        
        try:
            print(f"Populating vector database with {len(self.knowledge_base)} chunks...")
            
            documents = []
            metadatas = []
            ids = []
            
            for chunk in self.knowledge_base:
                documents.append(chunk['text'])
                ids.append(chunk['id'])
                
                # Prepare metadata
                metadata = chunk['metadata'].copy()
                for key, value in metadata.items():
                    if isinstance(value, list):
                        metadata[key] = ', '.join(map(str, value))
                    elif not isinstance(value, (str, int, float, bool, type(None))):
                        metadata[key] = str(value)
                
                metadatas.append(metadata)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            print(f"✅ Vector database populated with {len(documents)} chunks")
        except Exception as e:
            print(f"⚠️  Error populating vector DB: {e}")
    
    def retrieve_knowledge(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve relevant knowledge chunks."""
        if not self.collection or not self.embedding_model:
            return []
        
        try:
            if self.collection.count() == 0:
                return []
            
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count())
            )
            
            chunks = []
            for i in range(len(results['ids'][0])):
                chunks.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            return chunks
        except Exception as e:
            print(f"⚠️  Error retrieving knowledge: {e}")
            return []
    
    def generate_gpt4_response(self, question: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate response using GPT-4."""
        if not self.openai_client:
            return self._generate_basic_response(question, context_chunks)
        
        try:
            # Build context
            context_text = ""
            for i, chunk in enumerate(context_chunks, 1):
                context_text += f"\n--- Source {i} ---\n"
                context_text += f"Topic: {chunk['metadata'].get('topic', 'Unknown')}\n"
                context_text += f"Content: {chunk['text']}\n"
            
            system_prompt = """You are an expert space science educator. Provide accurate, 
engaging answers based on the provided context. Always cite your sources."""
            
            user_prompt = f"""Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            gpt_response = response.choices[0].message.content
            
            topics = list(set([chunk['metadata'].get('topic', 'Unknown') for chunk in context_chunks]))
            sources = [{
                'topic': chunk['metadata'].get('topic', 'Unknown'),
                'subtopic': chunk['metadata'].get('subtopic', 'Unknown')
            } for chunk in context_chunks]
            
            return {
                'response': gpt_response,
                'sources': sources,
                'topics': topics,
                'model_used': 'GPT-4'
            }
        except Exception as e:
            print(f"⚠️  GPT-4 error: {e}")
            return self._generate_basic_response(question, context_chunks)
    
    def _generate_basic_response(self, question: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Fallback response without GPT-4."""
        if not context_chunks:
            return {
                'response': "I don't have enough information to answer that question.",
                'sources': [],
                'topics': [],
                'model_used': 'Basic'
            }
        
        response_parts = []
        topics = set()
        
        for i, chunk in enumerate(context_chunks[:3], 1):
            topic = chunk['metadata'].get('topic', 'Space Science')
            topics.add(topic)
            response_parts.append(f"{i}. {chunk['text'][:200]}...")
        
        response = f"Based on space science knowledge:\n\n" + "\n\n".join(response_parts)
        
        sources = [{
            'topic': chunk['metadata'].get('topic', 'Unknown'),
            'subtopic': chunk['metadata'].get('subtopic', 'Unknown')
        } for chunk in context_chunks]
        
        return {
            'response': response,
            'sources': sources,
            'topics': list(topics),
            'model_used': 'Basic'
        }
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question."""
        self.conversation_history.append({
            'type': 'user',
            'content': question,
            'timestamp': datetime.now().isoformat()
        })
        
        # Optimize query
        optimized_query = self.query_optimizer.optimize_query(question)
        
        # Retrieve knowledge
        relevant_chunks = self.retrieve_knowledge(optimized_query['optimized_query'])
        
        # Generate response
        response_data = self.generate_gpt4_response(question, relevant_chunks)
        
        # Add to history
        self.conversation_history.append({
            'type': 'assistant',
            'content': response_data['response'],
            'timestamp': datetime.now().isoformat()
        })
        
        return response_data
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Compatibility method."""
        return self.ask(question)
    
    def get_available_topics(self) -> List[str]:
        """Get available topics."""
        if not self.collection:
            return []
        
        try:
            if self.collection.count() == 0:
                return []
            
            results = self.collection.get()
            topics = set()
            for metadata in results['metadatas']:
                if 'topic' in metadata:
                    topics.add(metadata['topic'])
            return sorted(list(topics))
        except Exception as e:
            print(f"⚠️  Error getting topics: {e}")
            return []
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def rebuild_knowledge_base(self, knowledge_base_path: str = "space_science_knowledge_base.json"):
        """Rebuild knowledge base."""
        if not self.chroma_client:
            print("⚠️  ChromaDB not available")
            return
        
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print("🔄 Deleted existing collection")
            
            self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
            
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Enhanced Space Science KB"}
            )
            
            self._populate_vector_db()
            print(f"✅ Knowledge base rebuilt with {len(self.knowledge_base)} chunks")
        except Exception as e:
            print(f"❌ Error rebuilding knowledge base: {e}")
