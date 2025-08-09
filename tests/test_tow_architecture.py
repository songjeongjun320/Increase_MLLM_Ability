"""
Unit tests for ToW architecture components.
"""

import pytest
import torch
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tow_architecture.core.tow_engine import ToWEngine
from tow_architecture.core.thought_processor import ThoughtTokenProcessor
from tow_architecture.core.cognitive_bridge import CognitiveBridge
from tow_architecture.core.multilingual_processor import MultilingualProcessor
from tow_architecture.utils.config import ToWConfig


class TestToWEngine:
    """Test cases for ToWEngine."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_adapter = Mock()
        self.mock_adapter.generate.return_value = "Test response"
        self.config = ToWConfig()
        
    def test_engine_initialization(self):
        """Test ToW engine initialization."""
        engine = ToWEngine(model_adapter=self.mock_adapter)
        assert engine.model_adapter == self.mock_adapter
        assert engine.thought_processor is not None
        assert engine.cognitive_bridge is not None
        assert engine.multilingual_processor is not None
    
    def test_process_basic_request(self):
        """Test basic processing request."""
        from tow_architecture.core.tow_engine import ToWRequest
        
        engine = ToWEngine(model_adapter=self.mock_adapter)
        request = ToWRequest(
            text="Hello world",
            target_language="ko",
            task_type="translation"
        )
        
        # Mock internal components
        engine.thought_processor.generate_thoughts = Mock(return_value={
            'thoughts': ['This is a greeting'],
            'metadata': {'confidence': 0.9}
        })
        
        response = engine.process(request)
        assert response is not None
        assert hasattr(response, 'output_text')
        assert hasattr(response, 'thought_tokens')


class TestThoughtTokenProcessor:
    """Test cases for ThoughtTokenProcessor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_adapter = Mock()
        self.processor = ThoughtTokenProcessor(model_adapter=self.mock_adapter)
    
    def test_thought_generation(self):
        """Test thought token generation."""
        self.mock_adapter.generate.return_value = "Thought: This is a test thought"
        
        thoughts = self.processor.generate_thoughts(
            text="Test input",
            target_language="ko",
            task_type="translation"
        )
        
        assert 'thoughts' in thoughts
        assert 'metadata' in thoughts
        assert isinstance(thoughts['thoughts'], list)
    
    def test_thought_types(self):
        """Test different thought types."""
        thought_types = self.processor._get_thought_types("translation")
        assert "TRANSLATION_ANALYSIS" in thought_types
        assert "CULTURAL_CONTEXT" in thought_types
    
    def test_quality_filtering(self):
        """Test thought quality filtering."""
        thoughts = [
            {"content": "This is a good thought", "confidence": 0.9},
            {"content": "bad", "confidence": 0.3},  # Should be filtered
            {"content": "Another good thought", "confidence": 0.8},
        ]
        
        filtered = self.processor._filter_by_quality(thoughts)
        assert len(filtered) == 2
        assert all(t["confidence"] >= 0.5 for t in filtered)


class TestCognitiveBridge:
    """Test cases for CognitiveBridge."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.bridge = CognitiveBridge()
    
    def test_bridge_thoughts(self):
        """Test cognitive bridging process."""
        thoughts = ["This is a greeting", "It should be polite"]
        target_language = "ko"
        task_type = "translation"
        
        result = self.bridge.bridge_thoughts(thoughts, target_language, task_type)
        
        assert 'semantic_mapping' in result
        assert 'cultural_adaptations' in result
        assert 'bridging_strategy' in result
    
    def test_cultural_adaptation(self):
        """Test cultural adaptation logic."""
        adaptations = self.bridge._get_cultural_adaptations("ko", "translation")
        assert isinstance(adaptations, list)
        assert any("formal_register" in str(a) or "honorific" in str(a) for a in adaptations)
    
    def test_semantic_mapping(self):
        """Test semantic mapping functionality."""
        thoughts = ["This is beautiful weather"]
        mapping = self.bridge._create_semantic_mapping(thoughts, "ko")
        
        assert isinstance(mapping, dict)
        # Should contain some mappings for common words


class TestMultilingualProcessor:
    """Test cases for MultilingualProcessor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_adapter = Mock()
        self.processor = MultilingualProcessor(model_adapter=self.mock_adapter)
    
    def test_output_generation(self):
        """Test multilingual output generation."""
        self.mock_adapter.generate.return_value = "안녕하세요"
        
        bridge_result = {
            'semantic_mapping': {'hello': '안녕'},
            'cultural_adaptations': ['formal_register']
        }
        
        output = self.processor.generate_output(
            bridge_result=bridge_result,
            target_language="ko",
            task_type="translation"
        )
        
        assert 'text' in output
        assert 'confidence' in output
        assert 'quality_metrics' in output
    
    def test_language_specific_processing(self):
        """Test language-specific post-processing."""
        # Test Korean honorifics
        text = "안녕하세요"
        processed = self.processor._apply_language_specific_processing(text, "ko")
        assert processed is not None
        
        # Test Chinese formality
        text = "你好"
        processed = self.processor._apply_language_specific_processing(text, "zh")
        assert processed is not None
    
    def test_quality_assessment(self):
        """Test output quality assessment."""
        output_text = "안녕하세요, 좋은 날씨네요"
        bridge_result = {
            'semantic_mapping': {'hello': '안녕', 'weather': '날씨'}
        }
        
        quality = self.processor._assess_quality(output_text, bridge_result, "ko")
        
        assert 'coherence' in quality
        assert 'cultural_appropriateness' in quality
        assert 'language_consistency' in quality


class TestIntegration:
    """Integration tests for complete ToW pipeline."""
    
    def setup_method(self):
        """Setup integration test fixtures."""
        self.mock_adapter = Mock()
        self.mock_adapter.generate.side_effect = [
            "Thought: This is a greeting in English",  # Thought generation
            "안녕하세요",  # Korean output
        ]
        
    def test_end_to_end_translation(self):
        """Test complete translation pipeline."""
        from tow_architecture.core.tow_engine import ToWRequest
        
        engine = ToWEngine(model_adapter=self.mock_adapter)
        
        request = ToWRequest(
            text="Hello, how are you?",
            target_language="ko",
            task_type="translation"
        )
        
        response = engine.process(request)
        
        assert response is not None
        assert response.output_text is not None
        assert len(response.thought_tokens) > 0
        assert response.confidence_score > 0
    
    def test_batch_processing(self):
        """Test batch processing capability."""
        from tow_architecture.core.tow_engine import ToWRequest
        
        engine = ToWEngine(model_adapter=self.mock_adapter)
        
        requests = [
            ToWRequest("Hello", "ko", "translation"),
            ToWRequest("Good morning", "ko", "translation"),
        ]
        
        responses = engine.process_batch(requests)
        
        assert len(responses) == 2
        assert all(r.output_text is not None for r in responses)


class TestConfiguration:
    """Test configuration management."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = ToWConfig()
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'evaluation')
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = ToWConfig()
        # Should not raise exceptions for valid config
        assert config.model.max_length > 0
        assert config.model.temperature >= 0
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = ToWConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'model' in config_dict
        assert 'training' in config_dict


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {
        'english_text': "The weather is beautiful today.",
        'korean_text': "오늘 날씨가 정말 좋네요.",
        'thoughts': [
            "The input describes positive weather conditions",
            "Korean translation requires appropriate formality level"
        ]
    }


def test_memory_management():
    """Test memory management for large models."""
    from tow_architecture.utils.memory_utils import MemoryManager
    
    if torch.cuda.is_available():
        manager = MemoryManager()
        initial_memory = manager.get_gpu_memory_usage()
        
        # Simulate memory allocation
        dummy_tensor = torch.randn(1000, 1000).cuda()
        increased_memory = manager.get_gpu_memory_usage()
        
        assert increased_memory > initial_memory
        
        # Cleanup
        del dummy_tensor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])