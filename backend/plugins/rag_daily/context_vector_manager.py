"""
Context Vector Manager for RAG Daily Plugin.

Manages conversation message vectors with decay aggregation and semantic segmentation.
Maintains a sliding window of context vectors with fuzzy matching capabilities.
"""

import numpy as np
import hashlib
from typing import Dict, List, Optional
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class ContextSegment:
    """A semantic segment of context with associated vectors."""

    def __init__(
        self,
        segment_id: str,
        messages: List[Dict],
        vector: np.ndarray,
        timestamp: float,
    ):
        self.segment_id = segment_id
        self.messages = messages
        self.vector = vector
        self.timestamp = timestamp

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "segment_id": self.segment_id,
            "messages": self.messages,
            "vector": self.vector.tolist() if isinstance(self.vector, np.ndarray) else self.vector,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ContextSegment":
        """Create from dictionary."""
        return cls(
            segment_id=data["segment_id"],
            messages=data["messages"],
            vector=np.array(data["vector"]),
            timestamp=data["timestamp"],
        )


class ContextVectorManager:
    """
    Manages context vectors for conversation messages.

    Features:
    - Decay-based vector aggregation
    - Semantic segmentation of message context
    - Fuzzy matching for context retrieval
    - Logic depth computation
    """

    def __init__(
        self,
        decay_rate: float = 0.75,
        max_context_window: int = 10,
        fuzzy_threshold: float = 0.85,
        dimension: int = 1024,
    ):
        """
        Initialize the context vector manager.

        Args:
            decay_rate: Decay rate for temporal aggregation (default: 0.75)
            max_context_window: Maximum number of context segments to keep
            fuzzy_threshold: Threshold for fuzzy matching (default: 0.85)
            dimension: Vector dimension (default: 1024 for bge-m3)
        """
        self.decay_rate = decay_rate
        self.max_context_window = max_context_window
        self.fuzzy_threshold = fuzzy_threshold
        self.dimension = dimension

        # Context storage
        self.segments: List[ContextSegment] = []
        self.message_vectors: Dict[str, np.ndarray] = {}  # message_id -> vector
        self.role_vectors: Dict[str, List[np.ndarray]] = {"user": [], "assistant": [], "system": []}

        # Timing
        self.last_update_time: Optional[float] = None

    def generate_hash(self, text):
        pass

    def update_context(
        self,
        messages: List[Dict],
        message_vectors: Optional[Dict[str, np.ndarray]] = None,
        allow_api: bool = False,
    ) -> None:
        """
        Update context vectors with new messages.

        Args:
            messages: List of message dictionaries with 'role', 'content', 'id' keys
            message_vectors: Optional pre-computed vectors for messages
            allow_api: Whether to allow API calls for missing vectors (not used in this implementation)
        """
        current_time = datetime.now().timestamp()
        self.last_update_time = current_time

        # Process new messages
        for msg in messages:
            msg_id = msg.get("id", f"msg_{current_time}_{len(self.message_vectors)}")

            if msg_id in self.message_vectors:
                continue  # Already processed

            # Store vector if provided
            if message_vectors and msg_id in message_vectors:
                vector = message_vectors[msg_id]
            else:
                # No vector provided, skip (embedding service should be called externally)
                continue

            self.message_vectors[msg_id] = vector

            # Store by role
            role = msg.get("role", "user")
            if role in self.role_vectors:
                self.role_vectors[role].append(vector)

        # Segment context if needed
        if len(messages) > 0:
            self.segment_context(messages)

        # Trim context window
        self._trim_context_window()

        logger.debug(f"[ContextVectorManager] Updated context with {len(messages)} messages")

    def segment_context(
        self,
        messages: List[Dict],
        threshold: float = 0.70,
    ) -> List[ContextSegment]:
        """
        Segment messages into semantic groups based on vector similarity.

        Args:
            messages: List of messages to segment
            threshold: Similarity threshold for segmentation (default: 0.70)

        Returns:
            List of context segments
        """
        if not messages:
            return []

        new_segments = []
        current_segment_messages = []
        current_segment_vectors = []

        for i, msg in enumerate(messages):
            msg_id = msg.get("id", f"msg_{i}")
            vector = self.message_vectors.get(msg_id)

            if vector is None:
                continue

            current_segment_messages.append(msg)
            current_segment_vectors.append(vector)

            # Check if we should segment (at least 2 messages in current segment)
            if len(current_segment_vectors) >= 2:
                # Calculate similarity with segment average
                segment_avg = np.mean(current_segment_vectors[:-1], axis=0)
                similarity = cosine_similarity(vector, segment_avg)

                if similarity < threshold:
                    # Create new segment
                    segment_id = f"seg_{len(self.segments) + len(new_segments)}_{datetime.now().timestamp()}"
                    segment_vector = np.mean(current_segment_vectors[:-1], axis=0)

                    segment = ContextSegment(
                        segment_id=segment_id,
                        messages=current_segment_messages[:-1].copy(),
                        vector=segment_vector,
                        timestamp=datetime.now().timestamp(),
                    )

                    new_segments.append(segment)

                    # Start new segment with current message
                    current_segment_messages = [msg]
                    current_segment_vectors = [vector]

        # Add the last segment
        if current_segment_messages:
            segment_id = f"seg_{len(self.segments) + len(new_segments)}_{datetime.now().timestamp()}"
            segment_vector = np.mean(current_segment_vectors, axis=0)

            segment = ContextSegment(
                segment_id=segment_id,
                messages=current_segment_messages,
                vector=segment_vector,
                timestamp=datetime.now().timestamp(),
            )

            new_segments.append(segment)

        # Merge new segments with existing
        self.segments.extend(new_segments)

        # Trim to max window
        if len(self.segments) > self.max_context_window:
            self.segments = self.segments[-self.max_context_window:]

        logger.debug(f"[ContextVectorManager] Created {len(new_segments)} new segments")

        return new_segments

    def get_context_summary(self) -> Dict:
        """
        Get a summary of the current context state.

        Returns:
            Dictionary with context statistics
        """
        return {
            "total_messages": len(self.message_vectors),
            "total_segments": len(self.segments),
            "role_counts": {
                role: len(vectors) for role, vectors in self.role_vectors.items()
            },
            "last_update": self.last_update_time,
        }

