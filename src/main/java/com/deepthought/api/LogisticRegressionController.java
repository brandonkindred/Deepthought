package com.qanairy.api;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import com.deepthought.models.LogisticRegressionModel;
import com.deepthought.models.repository.LogisticRegressionModelRepository;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;
import com.qanairy.brain.LogisticRegressionService;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Schema;

/**
 *	API endpoints for binary logistic regression. Supports training and predicting on raw
 *	 numeric feature matrices, plus a token-based convenience pair that decomposes JSON
 *	 documents via {@link com.qanairy.db.DataDecomposer} and one-hot encodes them against a
 *	 vocabulary persisted with the model.
 */
@RestController
@RequestMapping("/lr")
public class LogisticRegressionController {
	private final Logger log = LoggerFactory.getLogger(this.getClass());

	@Autowired
	private LogisticRegressionModelRepository lr_repo;

	@Autowired
	private LogisticRegressionService lr_service;

	@Operation(summary = "Train a binary logistic regression model on a numeric feature matrix",
			description = "", tags = { "Logistic Regression" })
	@RequestMapping(value = "/train", method = RequestMethod.POST)
	public @ResponseBody LogisticRegressionModel train(
			@Schema(description = "JSON-encoded feature matrix (rows of doubles)",
					example = "[[0,0],[0,1],[1,0],[2,2],[3,3]]", required = true)
			@RequestParam(value = "features", required = true) String features,
			@Schema(description = "JSON-encoded label vector of 0/1 (length must match feature row count)",
					example = "[0,0,0,1,1]", required = true)
			@RequestParam(value = "labels", required = true) String labels) {
		double[][] feature_matrix = parseFeatureMatrix(features);
		int[] label_vector = parseLabelVector(labels);
		log.debug("training logistic regression on {} samples x {} features",
				feature_matrix.length, feature_matrix.length > 0 ? feature_matrix[0].length : 0);
		return lr_service.train(feature_matrix, label_vector);
	}

	@Operation(summary = "Predict the class probability for a single feature vector",
			description = "", tags = { "Logistic Regression" })
	@RequestMapping(value = "/predict", method = RequestMethod.POST)
	public @ResponseBody Map<String, Object> predict(
			@Schema(description = "id of a previously trained LogisticRegressionModel",
					example = "1", required = true)
			@RequestParam(value = "model_id", required = true) long model_id,
			@Schema(description = "JSON-encoded feature vector (length must match the trained model)",
					example = "[2.5,2.5]", required = true)
			@RequestParam(value = "features", required = true) String features) {
		LogisticRegressionModel model = loadModel(model_id);
		double[] feature_vector = parseFeatureVector(features);
		double probability = lr_service.predictProbability(model, feature_vector);
		return buildPredictionResponse(probability, null);
	}

	@Operation(summary = "Train a binary logistic regression model from token-decomposed documents",
			description = "", tags = { "Logistic Regression" })
	@RequestMapping(value = "/train-from-tokens", method = RequestMethod.POST)
	public @ResponseBody LogisticRegressionModel trainFromTokens(
			@Schema(description = "JSON array of {input, label} samples; input may be JSON or plain text",
					example = "[{\"input\":\"submit button\",\"label\":1},{\"input\":\"footer\",\"label\":0}]",
					required = true)
			@RequestParam(value = "samples", required = true) String samples,
			@Schema(description = "Two display labels for class 0 and class 1 (informational only)",
					example = "noise,action", required = true)
			@RequestParam(value = "output_labels", required = true) String[] output_labels) {
		List<TokenSample> sample_list = parseSamples(samples);
		log.debug("training token-based logistic regression on {} samples", sample_list.size());
		return lr_service.trainFromTokens(sample_list, output_labels);
	}

	@Operation(summary = "Predict the class probability for a single token-decomposed input",
			description = "", tags = { "Logistic Regression" })
	@RequestMapping(value = "/predict-from-tokens", method = RequestMethod.POST)
	public @ResponseBody Map<String, Object> predictFromTokens(
			@Schema(description = "id of a model trained via /lr/train-from-tokens",
					example = "1", required = true)
			@RequestParam(value = "model_id", required = true) long model_id,
			@Schema(description = "JSON document or plain-text string to classify",
					example = "submit form", required = true)
			@RequestParam(value = "input", required = true) String input,
			@Schema(description = "Optional display labels for class 0 and class 1; if provided,"
					+ " the response will include predicted_label", example = "noise,action", required = false)
			@RequestParam(value = "output_labels", required = false) String[] output_labels) {
		LogisticRegressionModel model = loadModel(model_id);
		double probability = lr_service.predictProbabilityFromInput(model, input);
		return buildPredictionResponse(probability, output_labels);
	}

	private LogisticRegressionModel loadModel(long model_id) {
		Optional<LogisticRegressionModel> optional = lr_repo.findById(model_id);
		if (!optional.isPresent()) {
			throw new ResponseStatusException(HttpStatus.NOT_FOUND,
					"LogisticRegressionModel not found for id " + model_id);
		}
		return optional.get();
	}

	private Map<String, Object> buildPredictionResponse(double probability, String[] output_labels) {
		int predicted_class = lr_service.predictClass(probability);
		Map<String, Object> response = new HashMap<>();
		response.put("probability", probability);
		response.put("predicted_class", predicted_class);
		if (output_labels != null && output_labels.length == 2) {
			response.put("predicted_label", output_labels[predicted_class]);
		}
		return response;
	}

	private double[][] parseFeatureMatrix(String features) {
		try {
			Gson gson = new GsonBuilder().create();
			double[][] parsed = gson.fromJson(features, double[][].class);
			if (parsed == null) {
				throw new IllegalArgumentException("features must be a JSON-encoded 2D array");
			}
			return parsed;
		} catch (JsonSyntaxException e) {
			throw new IllegalArgumentException("features is not valid JSON", e);
		}
	}

	private double[] parseFeatureVector(String features) {
		try {
			Gson gson = new GsonBuilder().create();
			double[] parsed = gson.fromJson(features, double[].class);
			if (parsed == null) {
				throw new IllegalArgumentException("features must be a JSON-encoded 1D array");
			}
			return parsed;
		} catch (JsonSyntaxException e) {
			throw new IllegalArgumentException("features is not valid JSON", e);
		}
	}

	private int[] parseLabelVector(String labels) {
		try {
			Gson gson = new GsonBuilder().create();
			int[] parsed = gson.fromJson(labels, int[].class);
			if (parsed == null) {
				throw new IllegalArgumentException("labels must be a JSON-encoded 1D array");
			}
			return parsed;
		} catch (JsonSyntaxException e) {
			throw new IllegalArgumentException("labels is not valid JSON", e);
		}
	}

	private List<TokenSample> parseSamples(String samples) {
		try {
			Gson gson = new GsonBuilder().create();
			TokenSample[] parsed = gson.fromJson(samples, TokenSample[].class);
			if (parsed == null) {
				throw new IllegalArgumentException("samples must be a JSON-encoded array");
			}
			return Arrays.asList(parsed);
		} catch (JsonSyntaxException e) {
			throw new IllegalArgumentException("samples is not valid JSON", e);
		}
	}
}
