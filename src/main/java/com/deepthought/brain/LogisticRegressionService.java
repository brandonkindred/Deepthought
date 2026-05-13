package com.qanairy.brain;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.json.JSONException;
import org.json.JSONObject;
import org.springframework.stereotype.Component;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.sgd.linear.LinearSGDTrainer;
import org.tribuo.classification.sgd.objectives.LogMulticlass;
import org.tribuo.datasource.ListDataSource;
import org.tribuo.impl.ArrayExample;
import org.tribuo.math.optimisers.SGD;
import org.tribuo.provenance.SimpleDataSourceProvenance;

import com.deepthought.models.LogisticRegressionModel;
import com.deepthought.models.Token;
import com.deepthought.models.repository.LogisticRegressionModelRepository;
import com.qanairy.api.TokenSample;
import com.qanairy.db.DataDecomposer;

/**
 * Trains and evaluates binary logistic regression models. Wraps Tribuo's
 *  {@link LinearSGDTrainer} configured with a {@link LogMulticlass} objective (multinomial
 *  logistic regression) and persists the fitted {@link Model} on a
 *  {@link LogisticRegressionModel} node so a model can be referenced by id from a
 *  follow-up predict call.
 *
 * Predictions are computed by rehydrating the stored Tribuo model and calling
 *  {@link Model#predict(org.tribuo.Example)} so the model owns its own scoring math.
 */
@Component
public class LogisticRegressionService {

	private static final int DEFAULT_EPOCHS = 20;
	private static final double DEFAULT_LEARNING_RATE = 1.0;
	private static final String CLASS_ZERO = "0";
	private static final String CLASS_ONE = "1";

	private final LogisticRegressionModelRepository repo;

	public LogisticRegressionService(LogisticRegressionModelRepository repo) {
		this.repo = repo;
	}

	public LogisticRegressionModel train(double[][] features, int[] labels) {
		validateTrainingInputs(features, labels);
		return repo.save(fit(features, labels, numericFeatureNames(features[0].length)));
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

			List<Token> tokens = decompose(sample.getInput(), "sample input at index " + i);

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

		validateTrainingInputs(features, labels);
		String[] feature_names = vocabulary.toArray(new String[0]);
		LogisticRegressionModel model = fit(features, labels, feature_names);
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

		Model<Label> tribuo = model.getTribuoModel();
		if (tribuo == null) {
			throw new IllegalArgumentException("model has no persisted Tribuo state");
		}
		String[] names = featureNamesFor(model);
		return scoreClassOne(tribuo, names, features);
	}

	public double predictProbabilityFromInput(LogisticRegressionModel model, String input) {
		if (model == null) {
			throw new IllegalArgumentException("model cannot be null");
		}
		List<String> vocabulary = model.getVocabulary();
		if (vocabulary == null || vocabulary.isEmpty()) {
			throw new IllegalArgumentException("model was not trained from tokens; use the numeric predict endpoint");
		}

		List<Token> tokens = decompose(input, "input");
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

	private LogisticRegressionModel fit(double[][] features, int[] labels, String[] feature_names) {
		LabelFactory factory = new LabelFactory();
		List<org.tribuo.Example<Label>> examples = new ArrayList<>(features.length);
		for (int i = 0; i < features.length; i++) {
			Label label = new Label(labels[i] == 1 ? CLASS_ONE : CLASS_ZERO);
			examples.add(new ArrayExample<>(label, feature_names, features[i]));
		}

		ListDataSource<Label> source = new ListDataSource<>(examples, factory,
				new SimpleDataSourceProvenance("inline", factory));
		MutableDataset<Label> dataset = new MutableDataset<>(source);

		LinearSGDTrainer trainer = new LinearSGDTrainer(
				new LogMulticlass(),
				SGD.getLinearDecaySGD(DEFAULT_LEARNING_RATE),
				DEFAULT_EPOCHS,
				Trainer.DEFAULT_SEED);
		Model<Label> tribuoModel = trainer.train(dataset);

		LogisticRegressionModel model = new LogisticRegressionModel();
		model.setTribuoModel(tribuoModel);
		model.setNumFeatures(features[0].length);
		return model;
	}

	private double scoreClassOne(Model<Label> tribuo, String[] feature_names, double[] features) {
		ArrayExample<Label> example = new ArrayExample<>(new Label(CLASS_ZERO), feature_names, features);
		Prediction<Label> pred = tribuo.predict(example);
		Map<String, Label> scores = pred.getOutputScores();
		Label class_one = scores.get(CLASS_ONE);
		return class_one == null ? 0.0 : class_one.getScore();
	}

	private List<Token> decompose(String input, String context) {
		try {
			return DataDecomposer.decompose(new JSONObject(input));
		} catch (JSONException e) {
			try {
				return DataDecomposer.decompose(input);
			} catch (IllegalAccessException ex) {
				throw new IllegalArgumentException("failed to decompose " + context, ex);
			}
		} catch (IllegalAccessException e) {
			throw new IllegalArgumentException("failed to decompose " + context, e);
		}
	}

	private String[] featureNamesFor(LogisticRegressionModel model) {
		List<String> vocabulary = model.getVocabulary();
		if (vocabulary != null && !vocabulary.isEmpty()) {
			return vocabulary.toArray(new String[0]);
		}
		return numericFeatureNames(model.getNumFeatures());
	}

	private String[] numericFeatureNames(int n) {
		String[] names = new String[n];
		for (int i = 0; i < n; i++) {
			names[i] = "f" + i;
		}
		return names;
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
}
