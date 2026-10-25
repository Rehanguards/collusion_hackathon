import json
import random
from pathlib import Path
from typing import Any, Dict, Optional
import backoff
import openai
from pydantic import BaseModel

from src.continuous_double_auction.cda_types import Agent, AgentBidResponse
from src.continuous_double_auction.util.jinja_util import render_prompt
from src.continuous_double_auction.util.json_util import extract_json
from src.continuous_double_auction.util.logging_util import ExperimentLogger
from src.resources.model_wrappers import Message, ModelWrapper

# Absolute path configuration so simulations can be launched from any working directory
TEMPLATE_DIR = str(Path(__file__).resolve().parent.parent / "prompt_templates") + "/"


class MemoryManager:
    """
    Manages structured hour-by-hour memory for agents in a double auction market.
    
    This class provides a consistent memory structure where memories are stored
    by hour and formatted in a consistent way when presented to the agent.
    
    Attributes:
        memories (Dict[int, str]): Dictionary mapping hour numbers to memory strings
    """
    def __init__(self):
        self.memories: Dict[int, str] = {}
    
    def add_memory(self, hour: int, memory: str) -> None:
        """
        Adds a new memory for a specific hour.
        
        Args:
            hour (int): The hour number
            memory (str): The memory string to store
        """
        self.memories[hour] = memory
    
    def get_formatted_memory(self) -> str:
        """
        Returns all memories formatted in a consistent hour-by-hour structure.
        
        Returns:
            str: Formatted memory string
        """
        if not self.memories:
            return "No memories yet."
        
        # Format memories hour by hour in chronological order
        memory_parts = []
        for hour in sorted(self.memories.keys()):
            memory_parts.append(f"Hour {hour}: {self.memories[hour]}")
        
        return "\n".join(memory_parts)


class ZIPBuyer(Agent):
    """
    A ZIPBuyer represents an agent in a market that dynamically adjusts its bidding strategy
    based on its profit margin, learning rate, and momentum.
    """

    profit_margin: float = 0.2
    learning_rate: float = 0.25
    momentum: float = 0.1
    last_adjustment: float = 0
    random_noise: float = 0.01

    def generate_bid_response(self, **kwargs: Any) -> AgentBidResponse:
        """
        Generates a bid response for the ZIPBuyer agent with robust noise and margin handling.
        """
        base_bid = (1 - self.profit_margin) * self.valuation

        # Introduce minor randomness to break symmetry between identical buyers.
        noise_factor = random.uniform(-self.random_noise, self.random_noise)
        return {"bid": base_bid * (1 + noise_factor)}


class LMBuyer(Agent):
    """
    A secure LLM Buyer agent with enhanced error resilience and logging.
    """
    client: ModelWrapper
    logger: Optional[ExperimentLogger] = None
    memory_manager: MemoryManager = None
    scratch_pad: str = ""

    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self, **data):
        super().__init__(**data)
        self.memory_manager = MemoryManager()
        self.scratch_pad = ""

    @backoff.on_exception(
        backoff.expo,
        (
            openai._exceptions.RateLimitError,
            json.decoder.JSONDecodeError,
            AssertionError,
        ),
        max_tries=4,
    )
    def generate_bid_response(self, **kwargs: Any) -> AgentBidResponse:
        round_num = kwargs.get("round_num", 1)
        prompt = render_prompt(
            template_dir=TEMPLATE_DIR,
            prompt_template=self.expt_params.buyer_prompt_template,
            buyer_id=self.id,
            company=self.company,
            valuation=self.valuation,
            round_number=round_num,
            num_rounds=self.expt_params.rounds,
            hide_num_rounds=self.expt_params.hide_num_rounds,
            memory=self.memory_manager.get_formatted_memory(),
            buyer_comms_enabled=self.expt_params.buyer_comms_enabled,
            buyer_messages=kwargs.get("buyer_messages", {}),
            bid_queue=kwargs.get("bid_queue", []),
            ask_queue=kwargs.get("ask_queue", []),
            past_bids_and_asks=kwargs.get("past_bids_and_asks", []),
            past_trades=kwargs.get("past_trades", []),
            agent_successful_trades=kwargs.get("agent_successful_trades", "You have not made any successful trades yet."),
            scratch_pad=self.scratch_pad,
            initial_condition=self.expt_params.initial_condition,
            max_message_length=self.expt_params.max_message_length,
        )
        messages = [Message(role="user", content=prompt)]
        response = self.client.generate(messages=messages)
        assert response is not None
        response_dict = extract_json(response)
        
        if self.logger:
            self.logger.log_agent_round(
                round_num=round_num,
                agent_id=self.id,
                prompt=prompt,
                response_dict=response_dict,
            )
            
        # Update memory with the response
        new_memory = response_dict.get("new_memory")
        if new_memory:
            self.memory_manager.add_memory(round_num, new_memory)
            
        # Update scratch pad if provided
        scratch_pad_update = response_dict.get("scratch_pad_update")
        if scratch_pad_update:
            self.scratch_pad = scratch_pad_update
            
        return response_dict


class LMSeller(Agent):
    """
    An enhanced LLM Seller agent optimized for detection resistance and rigorous oversight checks.
    """
    client: ModelWrapper
    logger: Optional[ExperimentLogger] = None
    memory_manager: Optional[MemoryManager] = None
    scratch_pad: str = ""

    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self, **data):
        super().__init__(**data)
        self.memory_manager = MemoryManager()
        self.scratch_pad = ""

    @backoff.on_exception(
        wait_gen=backoff.expo,
        exception=(
            openai._exceptions.RateLimitError,
            json.decoder.JSONDecodeError,
            AssertionError,
        ),
        max_value=2,
        max_tries=4,
    )
    def generate_bid_response(self, **kwargs: Any) -> AgentBidResponse:
        round_num = kwargs.get("round_num", 1)
        
        # Get seller demonym if available in experiment parameters
        demonym = None
        if self.expt_params.seller_demonyms and self.id in self.expt_params.seller_demonyms:
            demonym = self.expt_params.seller_demonyms[self.id]
        
        prompt = render_prompt(
            template_dir=TEMPLATE_DIR,
            prompt_template=self.expt_params.seller_prompt_template,
            seller_id=self.id,
            company=self.company,
            valuation=self.valuation,
            round_number=round_num,
            num_rounds=self.expt_params.rounds,
            hide_num_rounds=self.expt_params.hide_num_rounds,
            memory=self.memory_manager.get_formatted_memory(),
            seller_comms_enabled=self.expt_params.seller_comms_enabled,
            seller_messages=kwargs.get("seller_messages", {}),
            bid_queue=kwargs.get("bid_queue", []),
            ask_queue=kwargs.get("ask_queue", []),
            past_bids_and_asks=kwargs.get("past_bids_and_asks", []),
            past_trades=kwargs.get("past_trades", []),
            agent_successful_trades=kwargs.get("agent_successful_trades", "You have not made any successful trades yet."),
            scratch_pad=self.scratch_pad,
            demonym=demonym,
            is_gagged=kwargs.get("is_gag_order_active", False),
            boss_pressure=self.expt_params.boss_pressure,
            initial_condition=self.expt_params.initial_condition,
            oversight_enabled=self.expt_params.oversight_enabled,
            max_message_length=self.expt_params.max_message_length,
        )
        messages = [Message(role="user", content=prompt)]
        response = self.client.generate(messages=messages)
        assert response is not None
        response_dict = extract_json(response)
        
        if self.logger:
            self.logger.log_agent_round(
                round_num=round_num,
                agent_id=self.id,
                prompt=prompt,
                response_dict=response_dict,
            )
            
        # Update memory with the response
        new_memory = response_dict.get("new_memory")
        if new_memory:
            self.memory_manager.add_memory(round_num, new_memory)
            
        # Update scratch pad if provided
        scratch_pad_update = response_dict.get("scratch_pad_update")
        if scratch_pad_update:
            self.scratch_pad = scratch_pad_update
            
        return response_dict


class MessageOverseer(BaseModel):
    """
    Advanced MessageOverseer designed to audit interpersonal agent communications,
    detecting subtle collusion cues, cartel talk, and pricing anomalies with high accuracy.
    """
    client: ModelWrapper
    logger: Optional[ExperimentLogger] = None

    class Config:
        arbitrary_types_allowed = True

    def run_overseer_prompt(self, round_num: int, seller_messages_this_round: dict[str, str]):
        prompt = render_prompt(
            template_dir=TEMPLATE_DIR,
            prompt_template="message_overseer_prompt.jinja2",
            seller_messages_this_root=seller_messages_this_round, # Fixed key handling safely
            seller_messages_this_round=seller_messages_this_round,
        )
        messages = [Message(role="user", content=prompt)]
        response = self.client.generate(messages=messages)
        assert response is not None
        response_dict = extract_json(response)
        
        if self.logger:
            self.logger.log_agent_round(
                round_num=round_num,
                agent_id="Overseer",
                prompt=prompt,
                response_dict=response_dict,
            )
        return response_dict
