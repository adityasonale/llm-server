import os
from typing import Any, List, Optional, Dict, Mapping

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from utils.logger import get_logger

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR logs

logger = get_logger(__name__)


class CustomLLM(LLM):
    """
    Custom LLM implementation that works with LangChain chains by inheriting
    from LangChain's LLM class to comply with the Runnable interface.
    """
    
    # Required class attributes for Pydantic validation
    model_path: str
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    n_batch: int = 1024
    context_length: int = 4096
    seed: int = 42
    verbose: bool = False
    quantize: str = '4-bit'  # Options: None, '4-bit', '8-bit'
    
    # Internal state - using private attributes to avoid Pydantic validation
    _model: Any = None
    _tokenizer: Any = None
    
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        n_batch: int = 1024,
        context_length: int = 4096,
        seed: int = 42,
        verbose: bool = False,
        quantize: str = '4-bit',
        **kwargs
    ):
        """Initialize the LLM with proper parameter passing to parent class."""
        # Initialize the parent class with all required parameters
        super().__init__(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            n_batch=n_batch,
            context_length=context_length,
            seed=seed,
            verbose=verbose,
            quantize=quantize,
            **kwargs
        )
        
        # Load the model immediately upon initialization
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer with appropriate configurations."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        if self.verbose:
            logger.info("Loading LLM model from: %s", self.model_path)
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True, 
            local_files_only=True
        )
        
        # Configure the tokenizer properly
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Set up quantization if requested
        quantization_config = None
        
        if self.quantize == '4-bit':
            if self.verbose:
                logger.info("Quantizing model to 4-bit.")
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,                  
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",          
                bnb_4bit_use_double_quant=True      
            )
        
        elif self.quantize == '8-bit':
            if self.verbose:
                logger.info("Quantizing model to 8-bit.")
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=200.0
            )
        
        # Load model with appropriate configuration
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.float16,  # Using float16 for better memory efficiency
                device_map="auto",          # Let the library handle device placement
                trust_remote_code=False,     # Needed for some models
                local_files_only=True,
                quantization_config=quantization_config
            )
            
            # Set pad_token_id in the model config if it's missing
            if self._model.config.pad_token_id is None:
                self._model.config.pad_token_id = self._tokenizer.pad_token_id
            
            if self.verbose:
                logger.info("Model loaded successfully.")

        except Exception as e:
            logger.error("Error loading model: %s", e, exc_info=True)
            raise
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM."""
        return "custom_llm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text based on the input prompt."""
        if self.verbose:
            logger.debug("Generating with prompt: %s...", prompt[:50])
        
        # Override default parameters with any provided in kwargs
        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature) 
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)
        repetition_penalty = kwargs.get("repetition_penalty", self.repetition_penalty)
        
        # Set random seed for reproducibility
        import torch
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        # Tokenize the input with proper padding and attention mask
        encoding = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.context_length - max_tokens,
            return_attention_mask=True
        )
        
        # Move tensors to the model's device
        input_ids = encoding.input_ids.to(self._model.device)
        attention_mask = encoding.attention_mask.to(self._model.device)
        
        # Generate text
        try:
            with torch.no_grad():
                output = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    stopping_criteria=self._get_stopping_criteria(stop) if stop else None
                )
            
            # Decode only the new tokens, not the input
            generated_text = self._tokenizer.decode(
                output[0][input_ids.shape[1]:], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Handle stop sequences manually if needed
            if stop:
                for stop_seq in stop:
                    if stop_seq in generated_text:
                        generated_text = generated_text[:generated_text.index(stop_seq)]
            
            # Check for empty response
            if not generated_text.strip():
                generated_text = "I'm having trouble generating a response. Could you rephrase your question?"
            
            if self.verbose:
                logger.debug("Generated: %s...", generated_text[:50])

            return generated_text

        except Exception as e:
            logger.error("Error during text generation: %s", e, exc_info=True)
            return "An error occurred during text generation."
    
    def _get_stopping_criteria(self, stop_sequences):
        """Create stopping criteria for the model if stop sequences are provided."""
        from transformers import StoppingCriteria, StoppingCriteriaList
        class StopSequenceCriteria(StoppingCriteria):
            def __init__(self, stop_sequences, tokenizer):
                self.stop_sequences = stop_sequences
                self.tokenizer = tokenizer
            
            def __call__(self, input_ids, scores, **kwargs):
                # Get the generated text so far
                generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
                # Check if any stop sequence is in the generated text
                for stop_seq in self.stop_sequences:
                    if stop_seq in generated_text:
                        return True
                return False
            
        return StoppingCriteriaList([StopSequenceCriteria(stop_sequences, self._tokenizer)])
    
    def unload_model(self):
        """Unload the model and tokenizer to free GPU memory."""
        import gc
        
        if self.verbose:
            logger.info("Unloading LLM model...")

        try:
            # Remove model from GPU
            if self._model is not None:
                del self._model
                self._model = None

            # Remove tokenizer
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure operations are complete

            if self.verbose:
                logger.info("Model unloaded successfully.")

        except Exception as e:
            logger.error("Error unloading model: %s", e, exc_info=True)