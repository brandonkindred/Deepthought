package com.qanairy.brain;

import java.util.ArrayList;
import java.util.List;

import org.json.JSONException;
import org.json.JSONObject;
import org.springframework.stereotype.Component;

import com.deepthought.models.LogisticRegressionModel;
import com.deepthought.models.Token;
import com.deepthought.models.repository.LogisticRegressionModelRepository;
import com.qanairy.api.TokenSample;
import com.qanairy.db.DataDecomposer;

import smile.classification.LogisticRegression;

/**
 * Trains and evaluates binary logistic regression models. Wraps Smile's
 *  {@link LogisticRegression#binomial(double[][], int[])} fitter and persists the
 *  resulting coefficients to a {@link LogisticRegressionModel} node so a model can be
 *  referenced by id from a follow-up predict call.
 *
 * Predictions are computed directly from the persisted intercept + weights via a sigmoid
 *  so we do not need to round-trip a Smile object through Neo4j.
 */
@Component
public class LogisticRegressionService {

	private final LogisticRegressionModelRepository repo;

	public LogisticRegressionService(LogisticRegressionModelRepository repo) {
		this.repo = repo;
	}

	public LogisticRegressionModel train(double[][] features, int[] labels) {
		return repo.save(fit(features, labels));
	}

	public LogisticRegressionModel trainFromTokens(List<TokenSample> samples, String[] output_labels) {
		if (samples == null || samples.size() < 2) {
			throw new IllegalArgumentException("training set must contain at least 2 samples");
		}
		if (output_labels == null || output_labels.length != 2) {
			throw new IllegalArgumentException("output_labels must contain exactly 2 entries (class 0 and class 1)");
		}

		List<List<String>> per_sample_tokens = new ArrayList<>(samples.size());
		List<String> vocabulary = new ArrayList<>();
		int[] labels = new int[samples.size()];

		for (int i = 0; i < samples.size(); i++) {
			TokenSample sample = samples.get(i);
			labels[i] = sample.getLabel();

			List<Token> tokens;
			try {
				tokens = DataDecomposer.decompose(new JSONObject(sample.getInput()));
			} catch (JSONException e) {
				try {
					tokens = DataDecomposer.decompose(sample.getInput());
				} catch (IllegalAccessException ex) {
					throw new IllegalArgumentException("failed to decompose sample input at index " + i, ex);
				}
			} catch (IllegalAccessException e) {
				throw new IllegalArgumentException("failed to decompose sample input at index " + i, e);
			}

			List<String> token_values = new ArrayList<>(tokens.size());
			for (Token token : tokens) {
				String value = token.getValue();
				if (value == null || value.trim().isEmpty()) {
					continue;
				}
				token_values.add(value);
				if (!vocabulary.contains(value)) {
					vocabulary.add(value);
				}
			}
			per_sample_tokens.add(token_values);
		}

		double[][] features = new double[samples.size()][vocabulary.size()];
		for (int i = 0; i < per_sample_tokens.size(); i++) {
			for (String value : per_sample_tokens.get(i)) {
				int col = vocabulary.indexOf(value);
				features[i][col] = 1.0;
			}
		}

		LogisticRegressionModel model = fit(features, labels);
		model.setVocabulary(vocabulary);
		return repo.save(model);
	}

	public double predictProbability(LogisticRegressionModel model, double[] features) {
		if (model == null) {
			throw new IllegalArgumentException("model cannot be null");
		}
		if (features == null) {
			throw new IllegalArgumentException("features cannot be null");
		}
		if (features.length != model.getNumFeatures()) {
			throw new IllegalArgumentException("feature vector length " + features.length
					+ " does not match trained model (" + model.getNumFeatures() + ")");
		}

		double[] weights = model.getWeights();
		double z = model.getIntercept();
		for (int i = 0; i < features.length; i++) {
			z += weights[i] * features[i];
		}
		return sigmoid(z);
	}

	public double predictProbabilityFromInput(LogisticRegressionModel model, String input) {
		if (model == null) {
			throw new IllegalArgumentException("model cannot be null");
		}
		List<String> vocabulary = model.getVocabulary();
		if (vocabulary == null || vocabulary.isEmpty()) {
			throw new IllegalArgumentException("model was not trained from tokens; use the numeric predict endpoint");
		}

		List<Token> tokens;
		try {
			tokens = DataDecomposer.decompose(new JSONObject(input));
		} catch (JSONException e) {
			try {
				tokens = DataDecomposer.decompose(input);
			} catch (IllegalAccessException ex) {
				throw new IllegalArgumentException("failed to decompose input", ex);
			}
		} catch (IllegalAccessException e) {
			throw new IllegalArgumentException("failed to decompose input", e);
		}

		double[] features = new double[vocabulary.size()];
		for (Token token : tokens) {
			String value = token.getValue();
			if (value == null) {
				continue;
			}
			int col = vocabulary.indexOf(value);
			if (col >= 0) {
				features[col] = 1.0;
			}
		}
		return predictProbability(model, features);
	}

	public int predictClass(double probability) {
		return probability >= 0.5 ? 1 : 0;
	}

	private LogisticRegressionModel fit(double[][] features, int[] labels) {
		validateTrainingInputs(features, labels);

		LogisticRegression.Binomial binomial = LogisticRegression.binomial(features, labels);
		double[] coefficients = binomial.coefficients();

		double[] weights = new double[coefficients.length - 1];
		System.arraycopy(coefficients, 0, weights, 0, weights.length);
		double intercept = coefficients[coefficients.length - 1];

		LogisticRegressionModel model = new LogisticRegressionModel();
		model.setIntercept(intercept);
		model.setWeights(weights);
		model.setNumFeatures(features[0].length);
		return model;
	}

	private void validateTrainingInputs(double[][] features, int[] labels) {
		if (features == null || labels == null) {
			throw new IllegalArgumentException("features and labels cannot be null");
		}
		if (features.length == 0) {
			throw new IllegalArgumentException("features cannot be empty");
		}
		if (features.length != labels.length) {
			throw new IllegalArgumentException("features row count " + features.length
					+ " does not match labels length " + labels.length);
		}
		if (features.length < 2) {
			throw new IllegalArgumentException("training set must contain at least 2 samples");
		}

		int first_class = labels[0];
		boolean has_other_class = false;
		for (int label : labels) {
			if (label != 0 && label != 1) {
				throw new IllegalArgumentException("labels must contain only 0 or 1 (got " + label + ")");
			}
			if (label != first_class) {
				has_other_class = true;
			}
		}
		if (!has_other_class) {
			throw new IllegalArgumentException("training set must contain at least one sample of each class");
		}

		int num_features = -1;
		for (int i = 0; i < features.length; i++) {
			if (features[i] == null) {
				throw new IllegalArgumentException("feature row " + i + " cannot be null");
			}
			if (num_features < 0) {
				num_features = features[i].length;
				if (num_features == 0) {
					throw new IllegalArgumentException("feature rows must be non-empty");
				}
			} else if (features[i].length != num_features) {
				throw new IllegalArgumentException("feature row " + i + " has length " + features[i].length
						+ " but expected " + num_features);
			}
		}
	}

	private static double sigmoid(double z) {
		return 1.0 / (1.0 + Math.exp(-z));
	}
}
