# Your component implementation:

"""Agent evaluates emotional state of other agents based on past interactions and adjusts behavior accordingly."""

from collections.abc import Callable, Mapping, Sequence
import datetime
import types

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component
from concordia.typing import logging

DEFAULT_PRE_ACT_KEY = 'Emotional Intelligence'

class EmotionalIntelligenceComponent(action_spec_ignored.ActionSpecIgnored):
    """Component that evaluates emotional states of other agents and adjusts behavior."""

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
        """Initializes the EmotionalIntelligenceComponent.

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

        # Retrieve past interactions from memory
        memory = self.get_entity().get_component(
            self._memory_component_name,
            type_=memory_component.MemoryComponent
        )
        recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
        memories = memory.retrieve(
            scoring_fn=recency_scorer, limit=self._num_memories_to_retrieve
        )

        # Extract interactions with other agents
        interactions = [
            mem.text for mem in memories
            if any(keyword in mem.text for keyword in ['interacted with', 'talked to', 'conversation with'])
        ]

        chain_of_thought = interactive_document.InteractiveDocument(self._model)

        interactions_text = '\n'.join(interactions)
        chain_of_thought.statement(f"Memories of {agent_name}'s interactions:\n{interactions_text}")
        chain_of_thought.statement(f'The current time: {self._clock_now()}.')

        # Analyze emotional states
        emotional_analysis = chain_of_thought.open_question(
            question=(
                f"Based on {agent_name}'s memories of past interactions, "
                "what are the likely emotional states of the other agents they have interacted with? "
                "Provide a brief summary for each agent, focusing on their perceived emotions."
            ),
            max_tokens=1000,
            terminators=(),
        )

        # Determine how to adjust behavior
        adjustment = chain_of_thought.open_question(
            question=(
                f"Given these perceived emotional states, how should {agent_name} adjust their behavior "
                "in future interactions to respond appropriately?"
            ),
            max_tokens=1000,
            terminators=(),
        )

        result = (
            f'[thought] {agent_name} recognizes the following emotional states in others: '
            f'{emotional_analysis}. '
            f'They plan to adjust their behavior accordingly: {adjustment}'
        )

        memory.add(result, metadata={})

        self._logging_channel({
            'Key': self.get_pre_act_key(),
            'Value': result,
            'Chain of thought': chain_of_thought.view().text().splitlines(),
        })

        return result
