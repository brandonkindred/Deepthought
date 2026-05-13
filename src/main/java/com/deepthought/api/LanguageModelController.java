package com.qanairy.api;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

import com.qanairy.brain.LanguageModelService;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Schema;

/**
 *	API endpoints that expose the current Neo4j graph as a bigram language model. Outgoing
 *	 {@code HAS_RELATED_TOKEN} edge weights are read live and normalized into a probability
 *	 distribution per call, so any weight updates from {@code /rl/learn} are reflected
 *	 immediately and nothing extra is persisted.
 */
@RestController
@RequestMapping("/llm")
public class LanguageModelController {
	private final Logger log = LoggerFactory.getLogger(this.getClass());

	@Autowired
	private LanguageModelService language_model_service;

	@Operation(summary = "Get the normalized next-token distribution for a seed token",
			description = "", tags = { "Language Model" })
	@RequestMapping(value = "/distribution", method = RequestMethod.GET)
	public @ResponseBody Map<String, Object> distribution(
			@Schema(description = "Seed token value to query outgoing weights for",
					example = "button", required = true)
			@RequestParam(value = "seed", required = true) String seed) {
		Map<String, Double> distribution = language_model_service.nextTokenDistribution(seed);
		Map<String, Object> response = new HashMap<>();
		response.put("seed", seed);
		response.put("distribution", distribution);
		return response;
	}

	@Operation(summary = "Predict the most probable next token after a seed",
			description = "", tags = { "Language Model" })
	@RequestMapping(value = "/predict-next", method = RequestMethod.POST)
	public @ResponseBody Map<String, Object> predictNext(
			@Schema(description = "Seed token value to predict from",
					example = "button", required = true)
			@RequestParam(value = "seed", required = true) String seed) {
		String next = language_model_service.predictNext(seed);
		Map<String, Object> response = new HashMap<>();
		response.put("seed", seed);
		response.put("next_token", next);
		return response;
	}

	@Operation(summary = "Generate a token sequence by walking the graph from a seed",
			description = "", tags = { "Language Model" })
	@RequestMapping(value = "/generate", method = RequestMethod.POST)
	public @ResponseBody Map<String, Object> generate(
			@Schema(description = "Seed token value to start generation from",
					example = "button", required = true)
			@RequestParam(value = "seed", required = true) String seed,
			@Schema(description = "Maximum number of tokens to emit (including the seed)",
					example = "10", required = false)
			@RequestParam(value = "max_length", required = false, defaultValue = "10") int max_length,
			@Schema(description = "If true, sample stochastically from each row; otherwise pick the argmax",
					example = "false", required = false)
			@RequestParam(value = "sample", required = false, defaultValue = "false") boolean sample,
			@Schema(description = "Optional PRNG seed for reproducible sampling (only used when sample=true)",
					example = "42", required = false)
			@RequestParam(value = "random_seed", required = false) Long random_seed) {
		Random random = sample ? (random_seed == null ? new Random() : new Random(random_seed)) : null;
		List<String> sequence = language_model_service.generate(seed, max_length, random);
		Map<String, Object> response = new HashMap<>();
		response.put("seed", seed);
		response.put("sequence", sequence);
		return response;
	}

	@ExceptionHandler(IllegalArgumentException.class)
	@ResponseStatus(HttpStatus.BAD_REQUEST)
	@ResponseBody
	public Map<String, String> handleIllegalArgument(IllegalArgumentException e) {
		log.debug("rejecting language model request: {}", e.getMessage());
		Map<String, String> body = new HashMap<>();
		body.put("error", e.getMessage());
		return body;
	}
}
