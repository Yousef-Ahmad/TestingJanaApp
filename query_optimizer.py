import re
import json
from typing import List, Dict

class SpaceScienceQueryOptimizer:
    def __init__(self):
        # Define topic categories and their related keywords
        self.topic_keywords = {
            "Mars Exploration": [
                "mars", "rover", "perseverance", "curiosity", "ingenuity", "helicopter",
                "jezero", "crater", "red planet", "martian", "sample", "drilling"
            ],
            "Planetary Science": [
                "planet", "planetary", "formation", "atmosphere", "atmospheric",
                "composition", "gas giant", "jupiter", "saturn", "exoplanet",
                "transit", "habitable", "solar system"
            ],
            "Space Missions": [
                "mission", "spacecraft", "probe", "satellite", "launch", "orbit",
                "voyager", "james webb", "jwst", "hubble", "iss", "space station",
                "exomars", "esa", "nasa"
            ],
            "Stellar Astronomy": [
                "star", "stellar", "sun", "solar", "binary", "supernova", "neutron",
                "formation", "evolution", "fusion", "nuclear", "luminosity",
                "magnitude", "constellation"
            ],
            "Space Phenomena": [
                "black hole", "gravitational wave", "solar wind", "space weather",
                "magnetosphere", "cosmic ray", "dark matter", "dark energy",
                "quasar", "pulsar", "galaxy", "nebula"
            ],
            "Latest Discoveries": [
                "discovery", "recent", "new", "breakthrough", "observation",
                "detection", "finding", "research", "study", "analysis",
                "2022", "2023", "2024"
            ]
        }
        
        # Mission-specific keywords
        self.mission_keywords = {
            "perseverance": ["mars 2020", "sample collection", "astrobiology", "moxie"],
            "curiosity": ["msl", "gale crater", "chemcam", "mahli"],
            "voyager": ["grand tour", "interstellar", "golden record"],
            "james webb": ["jwst", "infrared", "early universe", "exoplanet atmosphere"],
            "iss": ["international space station", "microgravity", "crew dragon"]
        }
        
        # Common synonyms and variations
        self.synonyms = {
            "red planet": "mars",
            "jwst": "james webb space telescope",
            "iss": "international space station",
            "exoplanet": "extrasolar planet",
            "alien world": "exoplanet",
            "space telescope": "telescope"
        }
    
    def normalize_query(self, query: str) -> str:
        """Normalize the query by converting to lowercase and handling synonyms."""
        query = query.lower().strip()
        
        # Replace synonyms
        for synonym, replacement in self.synonyms.items():
            query = query.replace(synonym, replacement)
        
        return query
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from the query."""
        normalized_query = self.normalize_query(query)
        keywords = []
        
        # Extract words
        words = re.findall(r'\b\w+\b', normalized_query)
        keywords.extend(words)
        
        # Look for multi-word phrases
        for topic, topic_words in self.topic_keywords.items():
            for keyword in topic_words:
                if keyword in normalized_query:
                    keywords.append(keyword)
        
        # Look for mission-specific phrases
        for mission, mission_words in self.mission_keywords.items():
            if mission in normalized_query:
                keywords.extend(mission_words)
        
        # Remove duplicates
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def identify_topics(self, query: str) -> List[str]:
        """Identify relevant topics based on the query content."""
        normalized_query = self.normalize_query(query)
        relevant_topics = []
        
        for topic, keywords in self.topic_keywords.items():
            topic_score = 0
            for keyword in keywords:
                if keyword in normalized_query:
                    topic_score += 1
            
            if topic_score > 0:
                relevant_topics.append((topic, topic_score))
        
        # Sort by relevance score
        relevant_topics.sort(key=lambda x: x[1], reverse=True)
        
        return [topic for topic, score in relevant_topics]
    
    def optimize_query(self, user_query: str) -> Dict:
        """Main function to optimize a user query."""
        if not user_query or user_query.strip() == "[insert user question here]":
            return {
                "error": "Please provide a valid question.",
                "optimized_query": user_query,
                "original_query": user_query,
                "extracted_keywords": [],
                "relevant_topics": []
            }
        
        # Extract components
        keywords = self.extract_keywords(user_query)
        topics = self.identify_topics(user_query)
        
        # Create optimized search query
        optimized_query = self.create_optimized_query(user_query, keywords, topics)
        
        return {
            "original_query": user_query,
            "optimized_query": optimized_query,
            "extracted_keywords": keywords[:10],
            "relevant_topics": topics[:3],
            "search_suggestions": self.generate_search_suggestions(keywords, topics)
        }
    
    def create_optimized_query(self, original_query: str, keywords: List[str], topics: List[str]) -> str:
        """Create an optimized version of the query."""
        optimized = self.normalize_query(original_query)
        
        # Add important keywords
        important_keywords = []
        for topic in topics[:2]:
            topic_keywords = self.topic_keywords.get(topic, [])
            for keyword in topic_keywords[:3]:
                if keyword not in optimized and keyword not in important_keywords:
                    important_keywords.append(keyword)
        
        if important_keywords:
            optimized += " " + " ".join(important_keywords)
        
        return optimized.strip()
    
    def generate_search_suggestions(self, keywords: List[str], topics: List[str]) -> List[str]:
        """Generate alternative search suggestions."""
        suggestions = []
        
        for topic in topics[:2]:
            suggestions.append(f"Search within {topic} category")
        
        if len(keywords) >= 2:
            suggestions.append(f"Combine keywords: {' + '.join(keywords[:3])}")
        
        mission_found = None
        for keyword in keywords:
            if keyword in self.mission_keywords:
                mission_found = keyword
                break
        
        if mission_found:
            suggestions.append(f"Focus on {mission_found.title()} mission details")
        
        return suggestions[:3]
