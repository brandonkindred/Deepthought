package com.qanairy.api;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
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
import org.springframework.web.server.ResponseStatusException;

import com.deepthought.models.GraphLanguageModel;
import com.deepthought.models.repository.GraphLanguageModelRepository;
import com.qanairy.brain.LanguageModelService;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Schema;

/**
 *	API endpoints for materializing a bigram language model from the current Neo4j edge
 *	 weights and generating token sequences from that model. {@code /llm/generate-model}
 *	 walks every {@code HAS_RELATED_TOKEN} edge once, normalizes the outgoing weights per
 *	 source token, and persists the resulting transition table as a
 *	 {@link GraphLanguageModel} that can be referenced by id from {@code /llm/generate}.
 */
@RestController
@RequestMapping("/llm")
public class LanguageModelController {
	private final Logger log = LoggerFactory.getLogger(this.getClass());

	@Autowired
	private GraphLanguageModelRepository model_repo;

	@Autowired
	private LanguageModelService language_model_service;

	@Operation(summary = "Generate a language model from the current Neo4j edge weights",
			description = "", tags = { "Language Model" })
	@RequestMapping(value = "/generate-model", method = RequestMethod.POST)
	public @ResponseBody GraphLanguageModel generateModel() {
		log.debug("generating language model snapshot from current graph weights");
		return language_model_service.generateModel();
	}

	@Operation(summary = "Retrieve a previously generated language model by id",
			description = "", tags = { "Language Model" })
	@RequestMapping(value = "/model", method = RequestMethod.GET)
	public @ResponseBody GraphLanguageModel getModel(
			@Schema(description = "id of a previously generated language model",
					example = "1", required = true)
			@RequestParam(value = "model_id", required = true) long model_id) {
		return loadModel(model_id);
	}

	@Operation(summary = "Predict the most probable next token after a seed token",
			description = "", tags = { "Language Model" })
	@RequestMapping(value = "/predict-next", method = RequestMethod.POST)
	public @ResponseBody Map<String, Object> predictNext(
			@Schema(description = "id of a previously generated language model",
					example = "1", required = true)
			@RequestParam(value = "model_id", required = true) long model_id,
			@Schema(description = "Seed token value to predict from",
					example = "button", required = true)
			@RequestParam(value = "seed", required = true) String seed) {
		GraphLanguageModel model = loadModel(model_id);
		String next = language_model_service.predictNext(model, seed);
		Map<String, Object> response = new HashMap<>();
		response.put("seed", seed);
		response.put("next_token", next);
		return response;
	}

	@Operation(summary = "Generate a token sequence from a seed using the persisted model",
			description = "", tags = { "Language Model" })
	@RequestMapping(value = "/generate", method = RequestMethod.POST)
	public @ResponseBody Map<String, Object> generate(
			@Schema(description = "id of a previously generated language model",
					example = "1", required = true)
			@RequestParam(value = "model_id", required = true) long model_id,
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
		GraphLanguageModel model = loadModel(model_id);
		Random random = sample ? (random_seed == null ? new Random() : new Random(random_seed)) : null;
		List<String> sequence = language_model_service.generate(model, seed, max_length, random);
		Map<String, Object> response = new HashMap<>();
		response.put("seed", seed);
		response.put("sequence", sequence);
		return response;
	}

	private GraphLanguageModel loadModel(long model_id) {
		Optional<GraphLanguageModel> optional = model_repo.findById(model_id);
		if (!optional.isPresent()) {
			throw new ResponseStatusException(HttpStatus.NOT_FOUND,
					"GraphLanguageModel not found for id " + model_id);
		}
		return optional.get();
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
