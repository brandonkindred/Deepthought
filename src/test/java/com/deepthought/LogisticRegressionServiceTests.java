package com.deepthought;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import com.deepthought.models.LogisticRegressionModel;
import com.deepthought.models.repository.LogisticRegressionModelRepository;
import com.deepthought.api.TokenSample;
import com.deepthought.brain.LogisticRegressionService;

@Test(groups = "Regression")
public class LogisticRegressionServiceTests {

	private LogisticRegressionModelRepository repo;
	private LogisticRegressionService service;

	@BeforeMethod
	public void setUp() {
		repo = mock(LogisticRegressionModelRepository.class);
		when(repo.save(any(LogisticRegressionModel.class)))
				.thenAnswer(invocation -> invocation.getArgument(0));
		service = new LogisticRegressionService(repo);
	}

	@Test
	public void trainsLinearlySeparableBinaryDataset() {
		double[][] features = new double[][] {
				{ 0.0, 0.0 }, { 0.1, 0.2 }, { 0.2, 0.3 }, { 0.3, 0.1 }, { 0.4, 0.4 },
				{ 2.0, 2.0 }, { 2.5, 2.8 }, { 3.0, 3.0 }, { 3.0, 2.0 }, { 2.0, 3.0 }
		};
		int[] labels = new int[] { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };

		LogisticRegressionModel model = service.train(features, labels);

		assertNotNull(model);
		assertEquals(model.getNumFeatures(), 2);
		assertNotNull(model.getTribuoModel());

		double prob_class_one = service.predictProbability(model, new double[] { 2.5, 2.5 });
		double prob_class_zero = service.predictProbability(model, new double[] { 0.0, 0.0 });
		assertTrue(prob_class_one > 0.5,
				"expected P(class=1 | [2.5, 2.5]) > 0.5, got " + prob_class_one);
		assertTrue(prob_class_zero < 0.5,
				"expected P(class=1 | [0, 0]) < 0.5, got " + prob_class_zero);
	}

	@Test
	public void predictProbabilityIsBoundedInZeroOne() {
		double[][] features = new double[][] {
				{ 0.0 }, { 0.1 }, { 0.2 }, { 5.0 }, { 5.1 }, { 5.2 }
		};
		int[] labels = new int[] { 0, 0, 0, 1, 1, 1 };

		LogisticRegressionModel model = service.train(features, labels);

		for (double x : new double[] { -100.0, -1.0, 0.0, 2.5, 100.0 }) {
			double p = service.predictProbability(model, new double[] { x });
			assertTrue(p >= 0.0 && p <= 1.0, "probability out of [0,1]: " + p + " for x=" + x);
		}
	}

	@Test
	public void predictClassReturnsOneAtOrAboveThreshold() {
		assertEquals(service.predictClass(0.5), 1);
		assertEquals(service.predictClass(0.49), 0);
		assertEquals(service.predictClass(0.99), 1);
		assertEquals(service.predictClass(0.0), 0);
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void rejectsAllSameLabel() {
		double[][] features = new double[][] { { 0.0 }, { 1.0 }, { 2.0 } };
		int[] labels = new int[] { 1, 1, 1 };
		service.train(features, labels);
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void rejectsMismatchedFeatureAndLabelCounts() {
		double[][] features = new double[][] { { 0.0 }, { 1.0 }, { 2.0 } };
		int[] labels = new int[] { 0, 1 };
		service.train(features, labels);
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void rejectsLabelsOutsideZeroOne() {
		double[][] features = new double[][] { { 0.0 }, { 1.0 } };
		int[] labels = new int[] { 0, 2 };
		service.train(features, labels);
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void rejectsNullFeatureRow() {
		double[][] features = new double[][] { null, { 1.0 } };
		int[] labels = new int[] { 0, 1 };
		service.train(features, labels);
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void predictRejectsMismatchedFeatureLength() {
		double[][] features = new double[][] { { 0.0, 0.0 }, { 5.0, 5.0 } };
		int[] labels = new int[] { 0, 1 };
		LogisticRegressionModel model = service.train(features, labels);
		service.predictProbability(model, new double[] { 1.0 });
	}

	@Test
	public void trainFromTokensBuildsVocabularyInInsertionOrder() {
		List<TokenSample> samples = new ArrayList<>();
		samples.add(new TokenSample("alpha beta", 0));
		samples.add(new TokenSample("gamma delta", 1));
		samples.add(new TokenSample("alpha gamma", 1));

		LogisticRegressionModel model = service.trainFromTokens(samples,
				new String[] { "noise", "action" });

		List<String> vocab = model.getVocabulary();
		assertNotNull(vocab);
		assertEquals(vocab, Arrays.asList("alpha", "beta", "gamma", "delta"));
		assertEquals(model.getNumFeatures(), vocab.size());
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void trainFromTokensRejectsWrongOutputLabelCount() {
		List<TokenSample> samples = new ArrayList<>();
		samples.add(new TokenSample("alpha", 0));
		samples.add(new TokenSample("beta", 1));
		service.trainFromTokens(samples, new String[] { "only_one" });
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void predictFromInputRejectsModelWithoutVocabulary() {
		double[][] features = new double[][] { { 0.0 }, { 1.0 } };
		int[] labels = new int[] { 0, 1 };
		LogisticRegressionModel model = service.train(features, labels);
		service.predictProbabilityFromInput(model, "anything");
	}
}
