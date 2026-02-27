from .base_processor import BaseProcessor
from .processors.deadlift import DeadliftProcessor

class ProcessorFactory:
    """
    Factory to create specific sport processors.
    """
    @staticmethod
    def get_processor(sport: str) -> BaseProcessor:
        sport = sport.lower()
        
        if sport == 'deadlift':
            return DeadliftProcessor()
        # Future extension:
        # elif sport == 'squat':
        #     return SquatProcessor()
        
        raise ValueError(f"No processor found for sport: {sport}")
