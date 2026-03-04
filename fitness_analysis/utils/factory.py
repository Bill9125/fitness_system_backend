from .base_processor import BaseProcessor
from .processors.deadlift import DeadliftProcessor
from .processors.benchpress import BenchpressProcessor

class ProcessorFactory:
    """
    Factory to create specific sport processors.
    """
    @staticmethod
    def get_processor(sport: str) -> BaseProcessor:
        sport = sport.lower()
        
        if sport == 'deadlift':
            return DeadliftProcessor()
        elif sport == 'benchpress':
            return BenchpressProcessor()
        
        raise ValueError(f"No processor found for sport: {sport}")
