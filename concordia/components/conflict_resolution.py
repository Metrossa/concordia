# Your ConflictResolutionComponent implementation:

"""Agent mediates conflicts by analyzing behaviors and proposing fair solutions."""

from collections.abc import Callable, Mapping
import datetime
import types

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component
from concordia.typing import logging

DEFAULT_PRE_ACT_KEY = 'Conflict Resolution'

class ConflictResolutionComponent(action_spec_ignored.ActionSpecIgnored):
    """Component that mediates conflicts and proposes compromises."""

    def __init__(
        self,
        model: language_model.LanguageModel,
        memory_component_name: str = (
            memory_component.DEFAULT_MEMORY_COMPONENT_NAME
        ),
        components: Mapping[
            entity_component.ComponentName, str
        ] = types.MappingProxyType({}),
        clock_now: Callable[[], datetime.datetime] | None = None,
        num_memories_to_retrieve: int = 100,
        pre_act_key: str = DEFAULT_PRE_ACT_KEY,
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        """Initializes the ConflictResolutionComponent.

        Args:
          model: The language model to use.
          memory_component_name: The name of the memory component from which to
            retrieve recent memories.
          components: The components to condition the answer on. This is a mapping
            of the component name to a label to use in the prompt.
          clock_now: Function that returns the current time.
          num_memories_to_retrieve: The number of memories to retrieve.
          pre_act_key: Prefix to add to the output of the component when called
            in `pre_act`.
          logging_channel: The channel to use for debug logging.
        """
        super().__init__(pre_act_key)
        self._model = model
        self._memory_component_name = memory_component_name
        self._components = dict(components)
        self._clock_now = clock_now or datetime.datetime.now
        self._num_memories_to_retrieve = num_memories_to_retrieve
        self._logging_channel = logging_channel

    def _make_pre_act_value(self) -> str:
        agent_name = self.get_entity().name

        # Retrieve past interactions involving conflicts
        memory = self.get_entity().get_component(
            self._memory_component_name,
            type_=memory_component.MemoryComponent
        )
        recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
        memories = memory.retrieve(
            scoring_fn=recency_scorer, limit=self._num_memories_to_retrieve
        )

        # Extract conflict-related interactions
        conflict_interactions = [
            mem.text for mem in memories
            if 'conflict' in mem.text.lower() or 'argument' in mem.text.lower()
            or 'disagreement' in mem.text.lower()
        ]

        if not conflict_interactions:
            return f'[thought] {agent_name} did not identify any conflicts to resolve at this time.'

        chain_of_thought = interactive_document.InteractiveDocument(self._model)

        conflict_text = '\n'.join(conflict_interactions)
        chain_of_thought.statement(f"Memories of conflicts involving {agent_name}:\n{conflict_text}")
        chain_of_thought.statement(f'The current time: {self._clock_now()}.')

        # Analyze conflicts and propose resolutions
        analysis = chain_of_thought.open_question(
            question=(
                f"Analyze these conflicts involving {agent_name}. "
                "Identify the main issues and the perspectives of each party involved."
            ),
            max_tokens=1000,
            terminators=(),
        )

        resolution = chain_of_thought.open_question(
            question=(
                f"Based on this analysis, how can {agent_name} mediate and propose fair solutions "
                "to resolve these conflicts? Suggest compromises or actions that promote cooperation."
            ),
            max_tokens=1000,
            terminators=(),
        )

        result = (
            f'[thought] {agent_name} analyzes the conflicts and proposes the following resolutions: '
            f'{resolution}'
        )

        memory.add(result, metadata={})

        self._logging_channel({
            'Key': self.get_pre_act_key(),
            'Value': result,
            'Chain of thought': chain_of_thought.view().text().splitlines(),
        })

        return result
