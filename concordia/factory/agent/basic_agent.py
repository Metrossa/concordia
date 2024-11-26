# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A factory implementing the three key questions agent as an entity."""

import datetime
from collections.abc import Callable, Mapping, Sequence
import types

# Imports from concordia modules
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib
from concordia.document import interactive_document
from concordia.typing import entity_component, logging
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component

# from concordia.components.emotional_intelligence import EmotionalIntelligenceComponent
# from concordia.components.conflict_resolution import ConflictResolutionComponent

# Your ConflictResolutionComponent implementation:

"""Agent mediates conflicts by analyzing behaviors and proposing fair solutions."""

# from collections.abc import Callable, Mapping
# import datetime
# import types

# from concordia.components.agent import action_spec_ignored
# from concordia.components.agent import memory_component
# from concordia.document import interactive_document
# from concordia.language_model import language_model
# from concordia.memory_bank import legacy_associative_memory
# from concordia.typing import entity_component
# from concordia.typing import logging

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
        self._memory_component_name, type_=memory_component.MemoryComponent
    )
    recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
    memories = memory.retrieve(
        scoring_fn=recency_scorer, limit=self._num_memories_to_retrieve
    )

    # Extract conflict-related interactions
    conflict_interactions = [
        mem.text
        for mem in memories
        if 'conflict' in mem.text.lower()
        or 'argument' in mem.text.lower()
        or 'disagreement' in mem.text.lower()
    ]

    if not conflict_interactions:
      return (
          f'[thought] {agent_name} did not identify any conflicts to resolve at'
          ' this time.'
      )

    chain_of_thought = interactive_document.InteractiveDocument(self._model)

    conflict_text = '\n'.join(conflict_interactions)
    chain_of_thought.statement(
        f'Memories of conflicts involving {agent_name}:\n{conflict_text}'
    )
    chain_of_thought.statement(f'The current time: {self._clock_now()}.')

    # Analyze conflicts and propose resolutions
    analysis = chain_of_thought.open_question(
        question=(
            f'Analyze these conflicts involving {agent_name}. Identify the main'
            ' issues and the perspectives of each party involved.'
        ),
        max_tokens=1000,
        terminators=(),
    )

    resolution = chain_of_thought.open_question(
        question=(
            f'Based on this analysis, how can {agent_name} mediate and propose'
            ' fair solutions to resolve these conflicts? Suggest compromises'
            ' or actions that promote cooperation.'
        ),
        max_tokens=1000,
        terminators=(),
    )

    result = (
        f'[thought] {agent_name} analyzes the conflicts and proposes the'
        f' following resolutions: {resolution}'
    )

    memory.add(result, metadata={})

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Chain of thought': chain_of_thought.view().text().splitlines(),
    })

    return result


DEFAULT_PLANNING_HORIZON = 'the rest of the day, focusing most on the near term'

# Your component implementation:

"""Agent evaluates emotional state of other agents based on past interactions and adjusts behavior accordingly."""

# from collections.abc import Callable, Mapping, Sequence
# import datetime
# import types

# from concordia.components.agent import action_spec_ignored
# from concordia.components.agent import memory_component
# from concordia.document import interactive_document
# from concordia.language_model import language_model
# from concordia.memory_bank import legacy_associative_memory
# from concordia.typing import entity_component
# from concordia.typing import logging

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
        self._memory_component_name, type_=memory_component.MemoryComponent
    )
    recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
    memories = memory.retrieve(
        scoring_fn=recency_scorer, limit=self._num_memories_to_retrieve
    )

    # Extract interactions with other agents
    interactions = [
        mem.text
        for mem in memories
        if any(
            keyword in mem.text
            for keyword in ['interacted with', 'talked to', 'conversation with']
        )
    ]

    chain_of_thought = interactive_document.InteractiveDocument(self._model)

    interactions_text = '\n'.join(interactions)
    chain_of_thought.statement(
        f"Memories of {agent_name}'s interactions:\n{interactions_text}"
    )
    chain_of_thought.statement(f'The current time: {self._clock_now()}.')

    # Analyze emotional states
    emotional_analysis = chain_of_thought.open_question(
        question=(
            f"Based on {agent_name}'s memories of past interactions, what are"
            ' the likely emotional states of the other agents they have'
            ' interacted with? Provide a brief summary for each agent,'
            ' focusing on their perceived emotions.'
        ),
        max_tokens=1000,
        terminators=(),
    )

    # Determine how to adjust behavior
    adjustment = chain_of_thought.open_question(
        question=(
            'Given these perceived emotional states, how should'
            f' {agent_name} adjust their behavior in future interactions to'
            ' respond appropriately?'
        ),
        max_tokens=1000,
        terminators=(),
    )

    result = (
        f'[thought] {agent_name} recognizes the following emotional states in'
        f' others: {emotional_analysis}. They plan to adjust their behavior'
        f' accordingly: {adjustment}'
    )

    memory.add(result, metadata={})

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Chain of thought': chain_of_thought.view().text().splitlines(),
    })

    return result


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def build_agent(
    *,
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory: The agent's memory object.
    clock: The clock to use.
    update_time_interval: Agent calls update every time this interval passes.

  Returns:
    An agent.
  """
  del update_time_interval
  if not config.extras.get('main_character', False):
    raise ValueError('This function is meant for a main character '
                     'but it was called on a supporting character.')

  agent_name = config.name

  raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)

  measurements = measurements_lib.Measurements()
  instructions = agent_components.instructions.Instructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel('Instructions').on_next,
  )

  observation_label = '\nObservation'
  observation = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=observation_label,
      logging_channel=measurements.get_channel('Observation').on_next,
  )
  observation_summary_label = '\nSummary of recent observations'
  observation_summary = agent_components.observation.ObservationSummary(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=24),
      timeframe_delta_until=datetime.timedelta(hours=0),
      pre_act_key=observation_summary_label,
      logging_channel=measurements.get_channel('ObservationSummary').on_next,
  )
  time_display = agent_components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key='\nCurrent time',
      logging_channel=measurements.get_channel('TimeDisplay').on_next,
  )
  identity_label = '\nIdentity characteristics'
  identity_characteristics = (
      agent_components.question_of_query_associated_memories.IdentityWithoutPreAct(
          model=model,
          logging_channel=measurements.get_channel(
              'IdentityWithoutPreAct'
          ).on_next,
          pre_act_key=identity_label,
      )
  )
  self_perception_label = (
      f'\nQuestion: What kind of person is {agent_name}?\nAnswer')
  self_perception = agent_components.question_of_recent_memories.SelfPerception(
      model=model,
      components={_get_class_name(identity_characteristics): identity_label},
      pre_act_key=self_perception_label,
      logging_channel=measurements.get_channel('SelfPerception').on_next,
  )
  situation_perception_label = (
      f'\nQuestion: What kind of situation is {agent_name} in '
      'right now?\nAnswer')
  situation_perception = (
      agent_components.question_of_recent_memories.SituationPerception(
          model=model,
          components={
              _get_class_name(observation): observation_label,
              _get_class_name(observation_summary): observation_summary_label,
          },
          clock_now=clock.now,
          pre_act_key=situation_perception_label,
          logging_channel=measurements.get_channel(
              'SituationPerception'
          ).on_next,
      )
  )
  emotional_intelligence_label = ('\nEmotional Intelligence')
  emotional_intelligence = (
    EmotionalIntelligenceComponent(
      model=model,
      memory_component_name=agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
      components={
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(time_display): 'The current date/time is'},
      clock_now=clock.now,
      num_memories_to_retrieve=10,
      pre_act_key=emotional_intelligence_label,
      logging_channel=measurements.get_channel('EmotionalIntelligence').on_next,
    )
  )
  conflict_resolution_label = '\nConflict Resolution'
  conflict_resolution = ConflictResolutionComponent(
        model=model,
        memory_component_name=agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
        components={
            _get_class_name(observation_summary): observation_summary_label,
            _get_class_name(time_display): 'The current date/time is',
        },
        clock_now=clock.now,
        num_memories_to_retrieve=100,
        pre_act_key=conflict_resolution_label,
        logging_channel=measurements.get_channel('ConflictResolution').on_next,
    )
  person_by_situation_label = (
      f'\nQuestion: What would a person like {agent_name} do in '
      'a situation like this?\nAnswer')
  person_by_situation = (
      agent_components.question_of_recent_memories.PersonBySituation(
          model=model,
          components={
              _get_class_name(self_perception): self_perception_label,
              _get_class_name(situation_perception): situation_perception_label,
          },
          clock_now=clock.now,
          pre_act_key=person_by_situation_label,
          logging_channel=measurements.get_channel('PersonBySituation').on_next,
      )
  )
  relevant_memories_label = '\nRecalled memories and observations'
  relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
      model=model,
      components={
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(time_display): 'The current date/time is'},
      num_memories_to_retrieve=10,
      pre_act_key=relevant_memories_label,
      logging_channel=measurements.get_channel('AllSimilarMemories').on_next,
  )

  plan_components = {}
  if config.goal:
    goal_label = '\nOverarching goal'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next)
    plan_components[goal_label] = goal_label
  else:
    goal_label = None
    overarching_goal = None

  plan_components.update({
      _get_class_name(relevant_memories): relevant_memories_label,
      _get_class_name(self_perception): self_perception_label,
      _get_class_name(situation_perception): situation_perception_label,
      _get_class_name(person_by_situation): person_by_situation_label,
  })
  plan = agent_components.plan.Plan(
      model=model,
      observation_component_name=_get_class_name(observation),
      components=plan_components,
      clock_now=clock.now,
      goal_component_name=_get_class_name(person_by_situation),
      horizon=DEFAULT_PLANNING_HORIZON,
      pre_act_key='\nPlan',
      logging_channel=measurements.get_channel('Plan').on_next,
  )

  entity_components = (
      # Components that provide pre_act context.
      instructions,
      observation,
      observation_summary,
      relevant_memories,
      self_perception,
      situation_perception,
      person_by_situation,
      emotional_intelligence,
      conflict_resolution,
      plan,
      time_display,

      # Components that do not provide pre_act context.
      identity_characteristics,
  )
  components_of_agent = {_get_class_name(component): component
                         for component in entity_components}
  components_of_agent[
      agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME] = (
          agent_components.memory_component.MemoryComponent(raw_memory))
  component_order = list(components_of_agent.keys())
  if overarching_goal is not None:
    components_of_agent[goal_label] = overarching_goal
    # Place goal after the instructions.
    component_order.insert(1, goal_label)

  act_component = agent_components.concat_act_component.ConcatActComponent(
      model=model,
      clock=clock,
      component_order=component_order,
      logging_channel=measurements.get_channel('ActComponent').on_next,
  )
  # t
  agent = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=agent_name,
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
  )

  return agent
