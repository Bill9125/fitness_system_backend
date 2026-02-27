from .factory import ProcessorFactory
from .processors.deadlift import DeadliftProcessor 

# Legacy support: Alias DeadliftProcessor so existing imports don't break immediately
# if someone imports directly, though logic suggests they should use factory.
